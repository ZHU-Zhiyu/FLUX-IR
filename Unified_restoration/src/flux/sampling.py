import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder

import numpy as np

def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    neg_image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1.0,
    neg_ip_scale: Tensor | float = 1.0
):
    i = 0
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            image_proj=image_proj,
            ip_scale=ip_scale, 
        )
        if i >= timestep_to_start_cfg:
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec, 
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale, 
            )     
            pred = neg_pred + true_gs * (pred - neg_pred)
        img = img + (t_prev - t_curr) * pred
        i += 1
    return img

def denoise_controlnet(
    model: Flux,
    controlnet:None,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    controlnet_gs=0.7,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    neg_image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1, 
    neg_ip_scale: Tensor | float = 1, 
):
    i = 0

    x_0 = img # noise
    x_1 = rearrange(controlnet_cond[1:2], "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    t_start = 1
    t = timesteps[t_start]
    img = (1 - t) * x_1 + t * x_0

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[t_start:], timesteps[t_start+1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        block_res_samples = controlnet(
                    img=img,
                    img_ids=img_ids,
                    controlnet_cond=controlnet_cond[0:1],
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            block_controlnet_hidden_states=[i * controlnet_gs for i in block_res_samples],
            image_proj=image_proj,
            ip_scale=ip_scale,
        )
        if i >= timestep_to_start_cfg:
            neg_block_res_samples = controlnet(
                        img=img,
                        img_ids=img_ids,
                        controlnet_cond=controlnet_cond[0:1],
                        txt=neg_txt,
                        txt_ids=neg_txt_ids,
                        y=neg_vec,
                        timesteps=t_vec,
                        guidance=guidance_vec,
                    )
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                block_controlnet_hidden_states=[i * controlnet_gs for i in neg_block_res_samples],
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale, 
            )     
            pred = neg_pred + true_gs * (pred - neg_pred)   
        img = img + (t_prev - t_curr) * pred

        i += 1
    return img

def denoise_controlnet_rein(
    model: Flux,
    controlnet:None,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    controlnet_gs=0.7,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    neg_image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1, 
    neg_ip_scale: Tensor | float = 1, 
    sde_step = 5,
    seed = 1024,
    sample_sde = True,
):
    # this is ignored for schnell
    i = 0
    # print(img.shape) #torch.Size([1, 4096, 64])
    # print(controlnet_cond.shape) #torch.Size([1, 16, 128, 128])

    # print(controlnet_cond.shape)

    X_ode_t = []
    X_sde_t = []

    x_0 = img # noise
    x_1 = rearrange(controlnet_cond[1:2], "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    t_start = 1
    t = timesteps[t_start]
    # print(t, x_0.shape, x_1.shape)
    img = (1 - t) * x_1 + t * x_0

    X_ode_t.append(img)
    X_sde_t.append(img)

    # sde_step = np.random.randint(1,9)
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[t_start:], timesteps[t_start+1:]):
        torch.cuda.empty_cache()
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        block_res_samples = controlnet(
                    img=img,
                    img_ids=img_ids,
                    controlnet_cond=controlnet_cond[0:1],
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            block_controlnet_hidden_states=[i * controlnet_gs for i in block_res_samples],
            image_proj=image_proj,
            ip_scale=ip_scale,
        )
        if i >= timestep_to_start_cfg and False:
            neg_block_res_samples = controlnet(
                        img=img,
                        img_ids=img_ids,
                        controlnet_cond=controlnet_cond[0:1],
                        txt=neg_txt,
                        txt_ids=neg_txt_ids,
                        y=neg_vec,
                        timesteps=t_vec,
                        guidance=guidance_vec,
                    )
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                block_controlnet_hidden_states=[i * controlnet_gs for i in neg_block_res_samples],
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale, 
            )     
            pred = neg_pred + true_gs * (pred - neg_pred)
   
        

        # sde sampling
        if i==sde_step and sample_sde:
            print('sde_step', sde_step)
            img = sde_sampling(t_curr=t_curr, deltaT=t_curr-t_prev, \
                               x_curr=img, pred_noise_residual=pred, seed=seed)
            img = img.to(torch.bfloat16)

        else:
            img = img + (t_prev - t_curr) * pred

        X_sde_t.append(img)
        i += 1
    return X_sde_t


def denoise_controlnet_samplesde(
    model: Flux,
    controlnet:None,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    controlnet_gs=0.7,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    neg_image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1, 
    neg_ip_scale: Tensor | float = 1, 
    sde_step = 8,
    seed = 1024,
    sample_sde = True,
):
    # this is ignored for schnell
    i = sde_step
    # print(img.shape) #torch.Size([1, 4096, 64])
    # print(controlnet_cond.shape) #torch.Size([1, 16, 128, 128])

    # print(controlnet_cond.shape)

    X_t = []

    t_start = sde_step

    X_t.append(img)

    # sde_step = np.random.randint(1,9)
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    # for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
    for t_curr, t_prev in zip(timesteps[t_start:], timesteps[t_start+1:]):
        # print(t_curr, t_prev)
        torch.cuda.empty_cache()
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        block_res_samples = controlnet(
                    img=img,
                    img_ids=img_ids,
                    controlnet_cond=controlnet_cond[0:1],
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            block_controlnet_hidden_states=[i * controlnet_gs for i in block_res_samples],
            image_proj=image_proj,
            ip_scale=ip_scale,
        )
        if i >= timestep_to_start_cfg and False:
            neg_block_res_samples = controlnet(
                        img=img,
                        img_ids=img_ids,
                        controlnet_cond=controlnet_cond[0:1],
                        txt=neg_txt,
                        txt_ids=neg_txt_ids,
                        y=neg_vec,
                        timesteps=t_vec,
                        guidance=guidance_vec,
                    )
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                block_controlnet_hidden_states=[i * controlnet_gs for i in neg_block_res_samples],
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale, 
            )     
            pred = neg_pred + true_gs * (pred - neg_pred)
   
        

        # sde sampling
        # if i<6 and i>1:
        if i==sde_step and sample_sde:
            print('sde_step', sde_step)
            img = sde_sampling(t_curr=t_curr, deltaT=t_curr-t_prev, \
                               x_curr=img, pred_noise_residual=pred, seed=seed)
            img = img.to(torch.bfloat16)
            X_t_sde = img

        else:
            img = img + (t_prev - t_curr) * pred
        
        X_t.append(img)
        i += 1
    return X_t, X_t_sde

def sde_sampling(t_curr, deltaT, x_curr, pred_noise_residual, seed):

    # pred = x0 - x1, x0 is noise, x1 is gt
    seed = np.random.randint(1, 10000)
    # print(seed)
    eplson = get_noise(
        1, 1024, 1024, device=x_curr.device,
        dtype=torch.float32, seed=seed
    )
    eplson = rearrange(eplson, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    alpha = torch.tensor([1+np.random.random()*1]).to(x_curr.device)
    # alpha = torch.tensor([np.random.randint(2,10)]).to(x_curr.device)

    beta_ =((t_curr - deltaT)**2 * (1 - (t_curr - alpha * deltaT))**2 / \
        (1 - (t_curr - deltaT))**2) - (t_curr - alpha * deltaT)**2
    beta = torch.sqrt(beta_)
    
    while beta_ < 0:
        alpha = torch.tensor([np.random.randint(2,10)]).to(x_curr.device)
        beta_ =((t_curr - deltaT)**2 * (1 - (t_curr - alpha * deltaT))**2 / \
                (1 - (t_curr - deltaT))**2) - (t_curr - alpha * deltaT)**2
        beta = torch.sqrt(beta_)

        if beta_ > 0:
            break
    
    # print(a)
    # print('alpha', alpha, 'beta', beta)
    # print(x_curr.shape, pred_noise_residual.shape, eplson.shape, alpha.shape)

    x_prev = (1 / ((1 + alpha * deltaT - t_curr) + torch.sqrt((t_curr - alpha * deltaT)**2 + beta**2))) * \
          (x_curr - alpha * deltaT * pred_noise_residual - beta * eplson)
    # print(x_prev.dtype)
    return x_prev

def denoise_controlnet_next_step(
    model: Flux,
    controlnet:None,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    controlnet_gs=0.7,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    neg_image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1, 
    neg_ip_scale: Tensor | float = 1,
    t_start = 1,
):
    # this is ignored for schnell
    i = 0
    # print(img.shape) #torch.Size([1, 4096, 64])
    # print(controlnet_cond.shape) #torch.Size([1, 16, 128, 128])

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    # for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
    # print(timesteps)
    for t_curr, t_prev in zip(timesteps[t_start:], timesteps[t_start+1:]):
        # print(t_curr, t_prev)
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        block_res_samples = controlnet(
                    img=img,
                    img_ids=img_ids,
                    controlnet_cond=controlnet_cond[0:1],
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            block_controlnet_hidden_states=[i * controlnet_gs for i in block_res_samples],
            image_proj=image_proj,
            ip_scale=ip_scale,
        )
        if i >= timestep_to_start_cfg and False:
            neg_block_res_samples = controlnet(
                        img=img,
                        img_ids=img_ids,
                        controlnet_cond=controlnet_cond[0:1],
                        txt=neg_txt,
                        txt_ids=neg_txt_ids,
                        y=neg_vec,
                        timesteps=t_vec,
                        guidance=guidance_vec,
                    )
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                block_controlnet_hidden_states=[i * controlnet_gs for i in neg_block_res_samples],
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale, 
            )     
            pred = neg_pred + true_gs * (pred - neg_pred)
   
        img = img + (t_prev - t_curr) * pred

        i += 1
        break
    return img, pred


def denoise_controlnet_next_step2(
    model: Flux,
    controlnet:None,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    controlnet_gs=0.7,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    neg_image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1, 
    neg_ip_scale: Tensor | float = 1,
    t_start = 1,
):
    # this is ignored for schnell
    i = 0
    # print(img.shape) #torch.Size([1, 4096, 64])
    # print(controlnet_cond.shape) #torch.Size([1, 16, 128, 128])

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    # for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
    # print(timesteps)
    for t_curr, t_prev in zip(timesteps[t_start:], timesteps[t_start+1:]):
        print(t_curr, t_prev)
        t_curr = t_prev + (t_curr - t_prev) * np.random.random()
        print(t_curr, t_prev)
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        block_res_samples = controlnet(
                    img=img,
                    img_ids=img_ids,
                    controlnet_cond=controlnet_cond[0:1],
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            block_controlnet_hidden_states=[i * controlnet_gs for i in block_res_samples],
            image_proj=image_proj,
            ip_scale=ip_scale,
        )
        if i >= timestep_to_start_cfg and False:
            neg_block_res_samples = controlnet(
                        img=img,
                        img_ids=img_ids,
                        controlnet_cond=controlnet_cond[0:1],
                        txt=neg_txt,
                        txt_ids=neg_txt_ids,
                        y=neg_vec,
                        timesteps=t_vec,
                        guidance=guidance_vec,
                    )
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                block_controlnet_hidden_states=[i * controlnet_gs for i in neg_block_res_samples],
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale, 
            )     
            pred = neg_pred + true_gs * (pred - neg_pred)
   
        img = img + (t_prev - t_curr) * pred

        i += 1
        break
    return img, pred

def denoise_controlnet_distill(
    model: Flux,
    controlnet:None,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    controlnet_gs=0.7,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    neg_image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1, 
    neg_ip_scale: Tensor | float = 1, 
    t_start = 1,
    t_jump_from_start = 0,
    t_end = 5
):
    # this is ignored for schnell
    i = 0
    # print(img.shape) #torch.Size([1, 4096, 64])
    # print(controlnet_cond.shape) #torch.Size([1, 16, 128, 128])

    # print(controlnet_cond.shape)

    x_0 = img # noise
    x_1 = rearrange(controlnet_cond[1:2], "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    t_start = 1
    t = timesteps[t_start]
    # print(t, x_0.shape, x_1.shape)
    img = (1 - t) * x_1 + t * x_0

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    # for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
    for t_curr, t_prev in zip(timesteps[t_start+t_jump_from_start:], timesteps[t_end:]):
        print(t_curr, t_prev)
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        block_res_samples = controlnet(
                    img=img,
                    img_ids=img_ids,
                    controlnet_cond=controlnet_cond[0:1],
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            block_controlnet_hidden_states=[i * controlnet_gs for i in block_res_samples],
            image_proj=image_proj,
            ip_scale=ip_scale,
        )
        if i >= timestep_to_start_cfg and False:
            neg_block_res_samples = controlnet(
                        img=img,
                        img_ids=img_ids,
                        controlnet_cond=controlnet_cond[0:1],
                        txt=neg_txt,
                        txt_ids=neg_txt_ids,
                        y=neg_vec,
                        timesteps=t_vec,
                        guidance=guidance_vec,
                    )
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                block_controlnet_hidden_states=[i * controlnet_gs for i in neg_block_res_samples],
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale, 
            )     
            pred = neg_pred + true_gs * (pred - neg_pred)
            # print(neg_pred)
   
        # if i==2:
        #     print('sde_step', 2)
        #     img = sde_sampling(t_curr=t_curr, deltaT=t_curr-t_prev, \
        #                        x_curr=img, pred_noise_residual=pred, seed=np.random.randint(1,1000))
        #     img = img.to(torch.bfloat16)
        # else:
        #     img = img + (t_prev - t_curr) * pred

        img = img + (t_prev - t_curr) * pred

        i += 1
        break
    return img, pred

def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )

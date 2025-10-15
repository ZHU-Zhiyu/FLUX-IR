import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import cv2
from torchvision import transforms
from sklearn.cluster import AgglomerativeClustering

transform = transforms.Lambda(lambda t: (t + 1) / 2)

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    # print(linear_start, linear_end, betas)
    # assert(1==2)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        self.loss_func_l2 = nn.MSELoss(reduction='sum').to(device)
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        # betas = np.clip(betas,a_min=0,a_max=0.6)
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))
        self.alphas_cumprod1 = np.append(1., alphas_cumprod)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise
    
    def calcu_cost(self, x, t, t_end, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            # x_recon = self.predict_start_from_noise(
            #     x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)[0])
            eplson = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)[0]

        # if clip_denoised:
        #     x_recon.clamp_(-1., 1.)

        x_t = x
        # cost = (x_t * (1-self.alphas_cumprod[t]).sqrt() - eplson) / (self.alphas_cumprod[t] * (1-self.alphas_cumprod[t])).sqrt()

        # if t == 9 and t_end == -1:
        #     cost = x_t / (self.alphas_cumprod[t]).sqrt() - \
        #         (0 - ((1/self.alphas_cumprod[t]).sqrt()) * (1-self.alphas_cumprod[t]).sqrt()) * eplson / \
        #         (self.alphas_cumprod[t].sqrt() - 1)           

        # if t == 0:
        #     cost = x_t / (self.alphas_cumprod[t]).sqrt() - \
        #         (0 - 0) * eplson / \
        #         (self.alphas_cumprod[t].sqrt() - 1)           
        # else:
        #     cost = x_t / (self.alphas_cumprod[t]).sqrt() - \
        #     ((1-self.alphas_cumprod[t_end]).sqrt() - ((self.alphas_cumprod[t_end]/self.alphas_cumprod[t]).sqrt()) * (1-self.alphas_cumprod[t]).sqrt()) * eplson / \
        #     (self.alphas_cumprod[t].sqrt() - self.alphas_cumprod[t_end].sqrt())

        # # cost = (x_t * (1-self.sqrt_alphas_cumprod_prev[t+1])- eplson) / (self.sqrt_alphas_cumprod_prev[t+1] * (1-self.alphas_cumprod[t]).sqrt())

        # # if t_end == 0:
        # #     cost = x_t / (self.sqrt_alphas_cumprod_prev[t+1]) - \
        # #         (0 - ((self.sqrt_alphas_cumprod_prev[t_end]/self.sqrt_alphas_cumprod_prev[t+1])) * \
        # #         (1-self.sqrt_alphas_cumprod_prev[t+1])) * eplson / \
        # #         (self.sqrt_alphas_cumprod_prev[t+1] - self.sqrt_alphas_cumprod_prev[t_end])


        # print(self.alphas_cumprod1)
        cost = x_t / np.sqrt(self.alphas_cumprod1[t+1]) - \
        (np.sqrt(1-self.alphas_cumprod1[t_end]) - (np.sqrt(self.alphas_cumprod1[t_end]/self.alphas_cumprod1[t+1])) * np.sqrt(1-self.alphas_cumprod1[t+1])) * eplson / \
        (np.sqrt(self.alphas_cumprod1[t+1]) - np.sqrt(self.alphas_cumprod1[t_end]))

        # cost = x_t / (self.alphas_cumprod1[t+1]) - \
        #     ((1-self.alphas_cumprod1[t_end]) - ((self.alphas_cumprod1[t_end]/self.alphas_cumprod1[t+1])) * \
        #     (1-self.alphas_cumprod1[t+1])) * eplson / \
        #     (self.alphas_cumprod1[t+1] - self.alphas_cumprod1[t_end])


        

        return cost

    def q_posterior_ode_add_noise(self, x_start, x_t, t):

        noise = torch.randn_like(x_start) 
        
        u = np.random.uniform(-0.11,0.11)
        s = np.sqrt(0.05)
        gamma = torch.normal(mean=u, std=0.01, size=(x_start.shape[0],x_start.shape[1],x_start.shape[2])).cuda()
        # gamma = 0.05

        eplson_coeff = ((1-self.alphas_cumprod[t]).sqrt()/(1-self.betas[t]).sqrt() - (1-self.alphas_cumprod_prev[t]).sqrt())
        eplson_coeff_new = eplson_coeff * (1 + gamma**2)

        posterior_mean  = (1/(1-self.betas[t]).sqrt()) *x_t - eplson_coeff_new * x_start \
            + gamma * (self.alphas_cumprod_prev[t] * torch.log(1/(1-self.betas[t]).sqrt())).sqrt() * noise

        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance_ode_addnoise(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            # x_recon = self.predict_start_from_noise(
            #     x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)[0])
            x_recon = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)[0]
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))
    
        # if clip_denoised:
        #     x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior_ode_add_noise(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance



    def q_posterior(self, x_start, x_t, t):

        aca =  (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        posterior_log_variance_clipped = (aca * self.betas[t]).sqrt()
        posterior_mean  = (1/(1-self.betas[t]).sqrt()) *x_t - ((1-self.alphas_cumprod[t]).sqrt()/(1-self.betas[t]).sqrt() - (1-self.alphas_cumprod_prev[t]).sqrt()) * x_start

        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped


    def q_posterior_ddpm(self, x_start, x_t, t):
        # ddpm_ori
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped


    def p_mean_variance_ddpm(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)[0])
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))
    
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior_ddpm(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance
    



    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        # print(t, self.sqrt_alphas_cumprod_prev[t+1], self.sqrt_alphas_cumprod_prev)
        # assert(1==2)
        if condition_x is not None:
            # x_recon = self.predict_start_from_noise(
            #     x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)[0])
            x_recon = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)[0]
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)

        return model_mean 


    @torch.no_grad()
    def p_sample_ddpm(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance_ddpm(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        return model_mean + noise * (0.5 * model_log_variance).exp()


    def p_sample_for_cost(self, x, t, t_end, condition_x=None):

        cost = self.calcu_cost(x=x, t=t, t_end=t_end, condition_x=condition_x)

        return cost

    def p_sample_for_ode(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, _ = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        
        return model_mean
    

    def p_sample_for_ode_addnoise(self, x, t, X_t_1, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance_ode_addnoise(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        
        variance = (0.5 * model_log_variance).exp()
        fx_t = model_mean

        pe_t_1 = (torch.mean(torch.log((1/(math.sqrt(2*math.pi) * variance)) * (-((X_t_1-fx_t)**2)/(2*variance**2)).exp()))).exp()

        return fx_t, pe_t_1

    def p_sample_loop_for_ode(self, x_in, t, continous=False):
        device = self.betas.device
        # sample_inter = (1 | (self.num_timesteps//10))
        sample_inter = 1
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in reversed(range(0, self.num_timesteps)):
                img, fx_t, p_t = self.p_sample_for_calcu_pe(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            shape = (1, 3, shape[2], shape[3])

            img = torch.randn(shape, device=device)
            # util.set_random_seed(seedt)
            ret_img = img
        
            for i in reversed(range(0, self.num_timesteps)):
                img = self.p_sample_for_ode(img, i, condition_x=x)
                # if i == self.num_timesteps-t-1:
                #     print(i, self.num_timesteps-t-1)
                #     img, pe_t_1 = self.p_sample_for_ode_addnoise(img, i, X_t_1, condition_x=x)
                # else:
                #     img, pe_t_1 = self.p_sample_for_ode(img, i, X_t_1, condition_x=x)
                # print(fx_t.shape, p_t)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        # ret_img = img
        # print('img', ret_img.shape)
        return ret_img

    @torch.no_grad()
    def p_sample_loop_for_calcu_cost(self, x_in, GT, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in reversed(range(0, self.num_timesteps)):
                img, fx_t, p_t = self.p_sample_for_calcu_pe(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            shape = (1, 3, shape[2], shape[3])
            b, c, h, w = x.shape
            
            img = torch.randn(shape, device=device)
            cost_10_to_t_all = []
            cost_t_to_0_all = []
            cost_all_T = []
            X = []
            X.append(img)
            for i in reversed(range(0, self.num_timesteps)):
                # print(i)
                img = self.p_sample(img, i, condition_x=x)
                X.append(img)
            X.reverse()

            cost = []
            cost_ = 0
            space = 6
            # print(self.alphas_cumprod1)
            for i in range(0, 8):
                # print(i, self.num_timesteps-space*i, self.num_timesteps-space*(i+1))
                Xt = X[self.num_timesteps-space*i]
                Xt_1 = X[self.num_timesteps-space*(i+1)]
                dxTdt = self.p_sample_for_cost(Xt, self.num_timesteps-space*i-1, self.num_timesteps-space*(i+1), condition_x=x)
                deltX_deltt_begin = (Xt -Xt_1) / (self.sqrt_alphas_cumprod_prev[self.num_timesteps-space*i]-self.sqrt_alphas_cumprod_prev[self.num_timesteps-space*(i+1)])
                # cost_ = str(((torch.sum((dxTdt - deltX_deltt_begin)**2)).sqrt()).cpu().item())[0:6]
                cost_temp = ((torch.sum((dxTdt - deltX_deltt_begin)**2)).sqrt()).cpu().item()
                cost_ = cost_ + cost_temp
                cost.append(cost_temp)
            
            # # print('--------------------------')
            # print(cost_all)
            for i in reversed(range(0, self.num_timesteps)):
                # print(i)
                XT = X[10]
                Xt = X[i]
                X0 = X[0]
                # dxTdt = self.p_sample_for_cost(XT, 9, condition_x=x)


                dxTdt = self.p_sample_for_cost(XT, 9, i, condition_x=x)
                dxtdt = self.p_sample_for_cost(Xt, i-1, 0, condition_x=x)
                
                deltX_deltt_begin = (XT -Xt) / (self.sqrt_alphas_cumprod_prev[10]-self.sqrt_alphas_cumprod_prev[i])
                deltX_deltt_end = (Xt - X0) / (self.sqrt_alphas_cumprod_prev[i]-self.sqrt_alphas_cumprod_prev[0])


                cost_10_to_t = str(((torch.sum((dxTdt - deltX_deltt_begin)**2)).sqrt()).cpu().item())[0:6]
                cost_t_to_0 = str(((torch.sum((dxtdt - deltX_deltt_end)**2)).sqrt()).cpu().item())[0:6]
  
                cost_10_to_t_all.append(cost_10_to_t)
                # cost_t_to_0 = str((torch.sum(cost_t_to_0)).cpu().item())[0:6]
                cost_t_to_0_all.append(cost_t_to_0)
                # cost_t_to_0_all.reverse()
                cost_all = float(cost_10_to_t) + float(cost_t_to_0)
                cost_all_T.append(str(cost_all)[0:6])
            
            # c1 = np.array(cost_10_to_t_all)
            # c2 = np.array(cost_t_to_0_all)
            # print(np.min(c1+c2), c1+c2)


            # print('T->t:', cost_10_to_t_all)
            # print('t->0:', cost_t_to_0_all)
            # print('Sum :', cost_all_T)

            return cost_10_to_t_all, cost_t_to_0_all
            

        if continous:
            return ret_img
        else:
            return img


    #################################### distill ##############################################

    def ode_jump_t(self, x_t, t_start, t_end, condition_x=None):
        x_in = condition_x
        condition_x = x_in['LQ']
        # print(condition_x.shape, x_t.shape)
        x_t = x_t.to(x_in['LQ'].device)
        batch_size = x_t.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t_start]]).repeat(batch_size, 1).to(x_t.device)
        if condition_x is not None:
            # x_recon = self.predict_start_from_noise(
            #     x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)[0])
            eplson = self.denoise_fn(torch.cat([condition_x, x_t], dim=1), noise_level)[0]
            
            shape = x_t.shape
            shape = (1, 3, shape[2], shape[3])
            noise_ = torch.randn(shape).to(x_t.device)
            noise = noise_
            for i in range(batch_size-1):
                noise = torch.cat([noise, noise_], dim=0)
            alpha_T = self.alphas_cumprod[t_start-1].sqrt()
            alpha_0 = self.alphas_cumprod[0].sqrt()
            eplson_wide = ((alpha_0/alpha_T) * noise - x_in['LQ']) / \
                        ((1-alpha_T ** 2).sqrt() * alpha_0 /alpha_T - (1 - alpha_0 ** 2).sqrt())
            omega = 0.001
            eplson_new = (1 + omega) * eplson - omega * eplson_wide

        posterior_mean  = (1/(1-self.betas[t_start-1]).sqrt()) * x_t - \
            ((1-self.alphas_cumprod[t_start-1]).sqrt()/(1-self.betas[t_start-1]).sqrt() - \
             (1-self.alphas_cumprod_prev[t_end]).sqrt()) * eplson_new

        return posterior_mean

    
    def p_sample_for_distill(self, x_in, x_t, t_start, t_end):

        img = self.ode_jump_t(x_t, t_start, t_end, condition_x=x_in)

        return img  

    #################################### distill ##############################################

    @torch.no_grad()
    def p_sample_for_reward(self, x, t, clip_denoised=True, condition_x=None):
        # model_mean, model_log_variance = self.p_mean_variance(
        #     x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x) # distilling
        model_mean, model_log_variance = self.p_mean_variance_ode_addnoise(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        variance = (0.5 * model_log_variance).exp()
        p_t = (torch.mean(torch.log((1/(math.sqrt(2*math.pi) * variance)) * (-0.5*noise**2).exp()))).exp()

        fx_t = model_mean

        x_t_1 = model_mean 

        return x_t_1, fx_t, p_t

    @torch.no_grad()
    def p_sample_loop_for_reward(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        sample_inter = 1
        # print(sample_inter)
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                # img, fx_t, p_t = self.p_sample_for_reward(img, i)
                img, fx_t, p_t = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            shape = (1, 3, shape[2], shape[3])

            img_ = torch.randn(shape, device=device)
            img = img_
            for i in range(x.shape[0]-1):
                img = torch.cat([img, img_], dim=0)
            # util.set_random_seed(seedt)
            # print(img)
            ret_img = img
            fx_t_list = []
            p_t_list = []
            # for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            for i in reversed(range(0, self.num_timesteps)):
                img, fx_t, p_t = self.p_sample_for_reward(img, i, condition_x=x)
                # util.set_random_seed(seedt[i])
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
                    fx_t_list.append(fx_t)
                    p_t_list.append(p_t)
        if continous:
            return ret_img, torch.cat(fx_t_list, dim=0), torch.as_tensor(p_t_list)
        else:
            return ret_img[-1]


    @torch.no_grad()
    def p_sample_loop_distill(self, x_in, NFE, dataset, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))

        x = x_in['LQ']
        shape = x.shape
        shape = (1, 3, shape[2], shape[3])
        # print(shape)
        # noise = torch.randn(shape, device=device)

        noise_ = torch.randn(shape, device=device)
        noise = noise_
        for i in range(x.shape[0]-1):
            noise = torch.cat([noise, noise_], dim=0)

        if dataset == 'LOLv1':

            img = torch.randn(shape, device=device)
            ret_img = img

            t_stride = 8 if NFE == 2 else 10
            t_end = self.num_timesteps-t_stride*1

            if t_stride == 10:
                img = self.p_sample_for_distill(x_in, img, t_start=self.num_timesteps, t_end=t_end)
                ret_img = torch.cat([ret_img, img], dim=0)
            elif t_stride == 8:
                img = self.p_sample_for_distill(x_in, img, t_start=self.num_timesteps, t_end=t_end)
                ret_img = torch.cat([ret_img, img], dim=0)                
                img = self.p_sample_for_distill(x_in, img, t_start=t_end, t_end=0)
                ret_img = torch.cat([ret_img, img], dim=0) 

        elif dataset == 'Raindrop':

            img = noise
            alpha = self.alphas_cumprod1[8]
            img = np.sqrt(alpha) * x_in['LQ'] + np.sqrt(1-alpha)*noise

            ret_img = img

            t_stride = 8 if NFE == 2 else 10
            t_end = self.num_timesteps-t_stride*1

            if t_stride == 10:
                img = self.p_sample_for_distill(x_in, img, t_start=self.num_timesteps-2, t_end=t_end)
                ret_img = torch.cat([ret_img, img], dim=0)
            elif t_stride == 8:
                img = self.p_sample_for_distill(x_in, img, t_start=self.num_timesteps-1, t_end=t_end)
                ret_img = torch.cat([ret_img, img], dim=0)                
                img = self.p_sample_for_distill(x_in, img, t_start=t_end, t_end=0)
                ret_img = torch.cat([ret_img, img], dim=0) 

        if continous:
            return ret_img
        else:
            return ret_img[-1]


    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)

            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            shape = (1, 3, shape[2], shape[3])
            img = torch.randn(shape, device=device)
            # util.set_random_seed(seedt)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution_for_reward(self, x_in, continous=False):
        return self.p_sample_loop_for_reward(x_in, continous)

    @torch.no_grad()
    def super_resolution_for_cost(self, x_in, GT, continous=False):
        return self.p_sample_loop_for_calcu_cost(x_in, GT, continous)
    
    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    @torch.no_grad()
    def super_resolution_distill(self, x_in, NFE, dataset, continous=False):
        return self.p_sample_loop_distill(x_in, NFE, dataset, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )


    def draw_features(self, x, savename):
        img = x[0, 0, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  
        img = img.astype(np.uint8)  
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imwrite(savename,img)


    def predict_start(self, x_t, continuous_sqrt_alpha_cumprod, noise):
        return (1. / continuous_sqrt_alpha_cumprod) * x_t - \
            (1. / continuous_sqrt_alpha_cumprod**2 - 1).sqrt() * noise

    def predict_t_minus1(self, x, t, continuous_sqrt_alpha_cumprod, noise, clip_denoised=True):

        x_recon = self.predict_start(x, 
                    continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), 
                    noise=noise)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, model_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        noise_z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        return model_mean + noise_z * (0.5 * model_log_variance).exp()       


    def to_patches(self, data, kernel_size):

        patches = nn.Unfold(kernel_size=kernel_size, stride=kernel_size)(torch.mean(data, dim=1, keepdim=True))
        patches = patches.transpose(2,1)
        return patches


    def calcu_kmeans(self, data, num_clusters):

        [b, h, w] = data.shape
        cluster_ids_all = np.empty([b, h])
        cluster_ids_all = torch.from_numpy(cluster_ids_all)
        for i in range(b):
            # cluster_ids, cluster_centers = kmeans(
            #     X=data[i,:,:], num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
            # )

            # DBSCAN
            # model = DBSCAN(eps=5)
            # cluster_ids = model.fit_predict(data[i,:,:].cpu())
            # cluster_ids = torch.from_numpy(cluster_ids).cuda()

            # # MeanShift
            # model = MeanShift()
            # cluster_ids = model.fit_predict(data[i,:,:].cpu())
            # cluster_ids = torch.from_numpy(cluster_ids).cuda()

            # # Spectral Clustering
            # model = SpectralClustering(n_clusters=num_clusters)
            # cluster_ids = model.fit_predict(transform(data[i,:,:].cpu()))
            # cluster_ids = torch.from_numpy(cluster_ids).cuda()

            # Hierarchical Clustering
            model = AgglomerativeClustering(n_clusters=num_clusters)
            cluster_ids = model.fit_predict(transform(data[i,:,:].cpu()))
            cluster_ids = torch.from_numpy(cluster_ids).cuda()

            # # gmm
            # model = GaussianMixture(n_components=num_clusters)
            # model.fit(data[i,:,:].cpu())
            # cluster_ids = model.predict(data[i,:,:].cpu())
            # cluster_ids = torch.from_numpy(cluster_ids).cuda()
            # print(cluster_ids)

            # # kmeans
            # km = kmeans_core(k=num_clusters,data_array=data[i,:,:].cpu().numpy(),batch_size=400,epochs=1000)
            # km.run()
            # cluster_ids = km.idx

            # print(cluster_ids)
            cluster_ids_all[i, :] = cluster_ids
        
        return cluster_ids_all

    def calcu_svd(self, data):

        u, sv, v = torch.svd(data)
        #sv_F2 = torch.norm(sv, dim=1)
        #sv_F2 = sv_F2.unsqueeze(1)
        #normalized_sv = sv / sv_F2

        return sv

    def calcu_svd_distance(self, data1, data2, cluster_ids, num_clusters):

        [b, h, w] = data1.shape 
        sv_ab_dis = np.empty([b, num_clusters])
        sv_ab_dis = torch.from_numpy(sv_ab_dis)
        for i in range(num_clusters):

            indices = (cluster_ids[0] ==i).nonzero(as_tuple=True)[0]
            
            if len(indices)==0:
                sv_ab_dis[:, i] = 1e-5
            else:
                data1_select = torch.index_select(data1, 1, indices.cuda())
                data2_select = torch.index_select(data2, 1, indices.cuda())
                sv1 = self.calcu_svd(data1_select.cpu())
                sv2 = self.calcu_svd(data2_select.cpu())
                sv_ab_dis_i = torch.abs(sv1 - sv2)
                sv_ab_dis[:, i] = torch.sum(sv_ab_dis_i, dim=1)
        return sv_ab_dis
    

    def calcu_dis_distance(self, data1, data2, cluster_ids, num_clusters, emb_net):

        [b, h, w] = data1.shape 
        sv_ab_dis = np.empty([b, num_clusters])
        sv_ab_dis = torch.from_numpy(sv_ab_dis)
        for i in range(num_clusters):

            indices = (cluster_ids[0] ==i).nonzero(as_tuple=True)[0]
            
            if len(indices)==0:
                sv_ab_dis[:, i] = 1e-5
            else:
                data1_select = torch.index_select(data1, 1, indices.cuda())
                data2_select = torch.index_select(data2, 1, indices.cuda())

                sv1 = emb_net(data1_select)
                sv2 = emb_net(data2_select)
                # sv1 = self.calcu_svd(data1_select.cpu())
                # print(sv1.shape)
                # batch, N, V = data1_select.shape
                # graph_data_list1 = []
                # graph_data_list2 = []
                # for j in range(batch):

                #     edge_index = torch.randint(0, N, (2, N))

                #     x1 = data1_select[j,:,:]
                #     gcndata1 = Data(x=x1, edge_index=edge_index)
                #     graph_data_list1.append(gcndata1)

                #     x2 = data2_select[j,:,:]
                #     gcndata2 = Data(x=x2, edge_index=edge_index)
                #     graph_data_list2.append(gcndata2)

                # loader1 = DataLoader(graph_data_list1, batch_size=batch)
                # loader2 = DataLoader(graph_data_list2, batch_size=batch)
                # for batch_data in loader1:
                #     sv1 = emb_net(batch_data.cuda())

                # for batch_data in loader2:
                #     sv2 = emb_net(batch_data.cuda())
                
                # # B, N, V = data1_select.shape
                # # batch = torch.arange(B).repeat_interleave(N)
                # # edge_index = torch.randint(0, N, (2, N))

                # # sv1 = emb_net(data1_select.reshape(B * N, V), edge_index.cuda(), batch.cuda())
                # # sv2 = emb_net(data2_select.reshape(B * N, V), edge_index.cuda(), batch.cuda())

                sv_ab_dis_i = torch.abs(sv1 - sv2)
                sv_ab_dis[:, i] = torch.sum(sv_ab_dis_i, dim=1)
        return sv_ab_dis    


    def reduce_mean(self, out_im, gt_im):
        return torch.abs(out_im - gt_im).mean()

    def perceptual_loss(self, img1, img2, vgg_model):
        output_1_vgg_0, output_1_vgg_1, output_1_vgg_2, output_1_vgg_3, output_1_vgg_4 = vgg_model(img1)
        output_2_vgg_0, output_2_vgg_1, output_2_vgg_2, output_2_vgg_3, output_2_vgg_4 = vgg_model(img2)
        loss_0 = self.reduce_mean(output_1_vgg_0, output_2_vgg_0)
        loss_1 = self.reduce_mean(output_1_vgg_1, output_2_vgg_1)
        loss_2 = self.reduce_mean(output_1_vgg_2, output_2_vgg_2)
        loss_3 = self.reduce_mean(output_1_vgg_3, output_2_vgg_3)
        loss_4 = self.reduce_mean(output_1_vgg_4, output_2_vgg_4)

        return loss_0 + loss_1 + loss_2 + loss_3 + loss_4

    def global_aware_losses(self, x_in, t=None, noise=None):

        x_start = x_in['GT']
        [b, c, h, w] = x_start.shape
        # print('t', t)
        if t == None:
            t = np.random.randint(1, self.num_timesteps + 1)

        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        # print(continuous_sqrt_alpha_cumprod)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            # with torch.no_grad():
            #     restored, transformer_features = pretrained_transformer(transform(x_in))
            x_recon, derained_img = self.denoise_fn(
                torch.cat([x_in['LQ'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        # loss_pix = self.loss_func(x_recon, noise)


        # # feature embedding
        # x_0 = x_start
        # x_0_patches = self.to_patches(x_0, kernel_size=8) 

        # x_t_1 = self.predict_t_minus1(x_noisy, t-1, 
        #         continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), 
        #         noise=x_recon) 
        # x_t_1_patches = self.to_patches(x_t_1, kernel_size=8)


        # cluster_ids = self.calcu_kmeans(x_0_patches, num_clusters=6)
        # svd_dis = self.calcu_dis_distance(x_0_patches, x_t_1_patches, cluster_ids=cluster_ids, num_clusters=6, emb_net=uct_model)
        # # print(t, continuous_sqrt_alpha_cumprod**4)
        # loss_emb = svd_dis.cuda()  * continuous_sqrt_alpha_cumprod**4
        # self.loss_func(x_recon, noise) +
        loss_pix = self.loss_func(derained_img, x_in['GT'])



        # ------------------- global structure ------------------- #
        # x_0 = x_start
        # x_0_patches = self.to_patches(x_0, kernel_size=8) 

        # x_t_1 = self.predict_t_minus1(x_noisy, t-1, 
        #         continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), 
        #         noise=x_recon) 
        # x_t_1_patches = self.to_patches(x_t_1, kernel_size=8)


        # cluster_ids = self.calcu_kmeans(x_0_patches, num_clusters=6)
        # svd_dis = self.calcu_svd_distance(x_0_patches, x_t_1_patches, cluster_ids=cluster_ids, num_clusters=6)


        # lambda_ = 10
        # loss_pix = self.loss_func(x_recon, noise) * lambda_

        # loss_s = svd_dis.cuda() * continuous_sqrt_alpha_cumprod**4

        # --------------------- perceptual_loss --------------------- #
        # x_0 = x_start
        # x_t_1 = self.predict_t_minus1(x_noisy, t-1, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=x_recon) 
        
        # loss_pix = self.loss_func(x_recon, noise)
        # p_loss = self.perceptual_loss(transform(x_t_1.cpu()).cuda(), transform(x_0.cpu()).cuda(), vgg_model)
        # loss_s = p_loss # * continuous_sqrt_alpha_cumprod**4


        # -------------------- L1 ||x_t_1 - x0|| -------------------- #
        # loss_pix = self.loss_func(x_recon, noise) 
        # svd_dis = self.loss_func(x_t_1, x_0)
        # print(t, loss_pix, svd_dis)
        # loss_s = svd_dis #* continuous_sqrt_alpha_cumprod**4


        return loss_pix

    def forward(self, x, *args, **kwargs):
        return self.global_aware_losses(x, *args, **kwargs)

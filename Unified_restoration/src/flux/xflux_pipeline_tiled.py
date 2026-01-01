from PIL import Image, ExifTags
import numpy as np
import torch
from torch import Tensor

from einops import rearrange
import uuid
import os
from torchvision import transforms
import torch.nn.functional as F

from src.flux.modules.layers import (
    SingleStreamBlockProcessor,
    DoubleStreamBlockLoraProcessor,
    IPDoubleStreamBlockProcessor,
    ImageProjModel,
)
from src.flux.sampling import denoise, denoise_controlnet, denoise_controlnet_rein, get_noise, get_schedule, prepare, unpack
from src.flux.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    load_controlnet,
    load_flow_model_quintized,
    Annotator,
    get_lora_rank,
    load_checkpoint
)

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

class XFluxPipeline:
    def __init__(self, model_type, device, offload: bool = False):
        self.device = torch.device(device)
        self.offload = offload
        self.model_type = model_type

        self.clip = load_clip(self.device)
        self.t5 = load_t5(self.device, max_length=512)
        self.ae = load_ae(model_type, device="cpu" if offload else self.device)
        
        self.ae2 = load_ae(model_type, device="cpu" if offload else self.device)
        self.ae2.encoder.load_state_dict(torch.load('experiments/uir_vae_encoder/checkpoint-2600/encoder_lq.bin'))

        if "fp8" in model_type:
            self.model = load_flow_model_quintized(model_type, device="cpu" if offload else self.device)
        else:
            self.model = load_flow_model(model_type, device="cpu" if offload else self.device)

        self.image_encoder_path = "openai/clip-vit-large-patch14"
        self.hf_lora_collection = "XLabs-AI/flux-lora-collection"
        self.lora_types_to_names = {
            "realism": "lora.safetensors",
        }
        self.controlnet_loaded = False
        self.ip_loaded = False

    def set_ip(self, local_path: str = None, repo_id = None, name: str = None):
        self.model.to(self.device)

        # unpack checkpoint
        checkpoint = load_checkpoint(local_path, repo_id, name)
        prefix = "double_blocks."
        blocks = {}
        proj = {}

        for key, value in checkpoint.items():
            if key.startswith(prefix):
                blocks[key[len(prefix):].replace('.processor.', '.')] = value
            if key.startswith("ip_adapter_proj_model"):
                proj[key[len("ip_adapter_proj_model."):]] = value

        for key, value in checkpoint.items():
            if key.startswith(prefix):
                blocks[key[len(prefix):].replace('.processor.', '.')] = value
            if key.startswith("ip_adapter_proj_model"):
                proj[key[len("ip_adapter_proj_model."):]] = value

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()

        # setup image embedding projection model
        self.improj = ImageProjModel(4096, 768, 4)
        self.improj.load_state_dict(proj)
        self.improj = self.improj.to(self.device, dtype=torch.bfloat16)

        ip_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            ip_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    ip_state_dict[k.replace(f'{name}.', '')] = checkpoint[k]
            if ip_state_dict:
                ip_attn_procs[name] = IPDoubleStreamBlockProcessor(4096, 3072)
                ip_attn_procs[name].load_state_dict(ip_state_dict)
                ip_attn_procs[name].to(self.device, dtype=torch.bfloat16)
            else:
                ip_attn_procs[name] = self.model.attn_processors[name]

        self.model.set_attn_processor(ip_attn_procs)
        self.ip_loaded = True

    def set_lora(self, local_path: str = None, repo_id: str = None,
                 name: str = None, lora_weight: int = 0.7):
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.update_model_with_lora(checkpoint, lora_weight)

    def set_lora_from_collection(self, lora_type: str = "realism", lora_weight: int = 0.7):
        checkpoint = load_checkpoint(
            None, self.hf_lora_collection, self.lora_types_to_names[lora_type]
        )
        self.update_model_with_lora(checkpoint, lora_weight)

    def update_model_with_lora(self, checkpoint, lora_weight):
        rank = get_lora_rank(checkpoint)
        lora_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            if name.startswith("single_blocks"):
                lora_attn_procs[name] = SingleStreamBlockProcessor()
                continue
            lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=rank)
            lora_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    lora_state_dict[k[len(name) + 1:]] = checkpoint[k] * lora_weight
            lora_attn_procs[name].load_state_dict(lora_state_dict)
            lora_attn_procs[name].to(self.device)

        self.model.set_attn_processor(lora_attn_procs)

    def set_controlnet(self, control_type: str, local_path: str = None, repo_id: str = None, name: str = None):
        self.model.to(self.device)
        self.controlnet = load_controlnet(self.model_type, self.device).to(torch.bfloat16)

        checkpoint = load_checkpoint(local_path, repo_id, name)
        print(local_path, repo_id, name)
        self.controlnet.load_state_dict(checkpoint, strict=False)
        self.annotator = Annotator(control_type, self.device)
        self.controlnet_loaded = True
        self.control_type = control_type

    def get_image_proj(
        self,
        image_prompt: Tensor,
    ):
        # encode image-prompt embeds
        image_prompt = self.clip_image_processor(
            images=image_prompt,
            return_tensors="pt"
        ).pixel_values
        image_prompt = image_prompt.to(self.image_encoder.device)
        image_prompt_embeds = self.image_encoder(
            image_prompt
        ).image_embeds.to(
            device=self.device, dtype=torch.bfloat16,
        )
        # encode image
        image_proj = self.improj(image_prompt_embeds)
        return image_proj

    # tiled vae
    def split_with_overlap(self, latent, tile_size, overlap_h, overlap_w):
        """
        使用水平和垂直 overlap 对 latent 进行分块。
        Args:
            latent: Tensor, (B, C, H, W)
            tile_size: int, 每个分块的大小 (tile_size x tile_size)
            overlap_h: int, 垂直方向的重叠大小
            overlap_w: int, 水平方向的重叠大小
        Returns:
            tiles: List[Tensor], 所有的分块
            positions: List[Tuple[int, int]], 每个分块的 (y, x) 起始位置
        """
        _, _, H, W = latent.shape
        stride_h = tile_size - overlap_h
        stride_w = tile_size - overlap_w

        tiles = []
        positions = []

        # 滑动窗口分块
        for y in range(0, H - overlap_h, stride_h):
            for x in range(0, W - overlap_w, stride_w):
                # 确保最后的 tile 能覆盖完整区域
                y_end = min(y + tile_size, H)
                x_end = min(x + tile_size, W)
                tile = latent[:, :, y:y_end, x:x_end]

                # 如果 tile 尺寸不足，进行 padding
                if tile.size(-2) < tile_size or tile.size(-1) < tile_size:
                    pad_h = tile_size - tile.size(-2)
                    pad_w = tile_size - tile.size(-1)
                    tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h))

                tiles.append(tile)
                positions.append((y, x))

        return tiles, positions


    def merge_with_overlap(self, tiles, positions, latent_shape, tile_size, overlap_h, overlap_w):
        """
        拼接分块后的 latent 表示。
        Args:
            tiles: List[Tensor], 所有的分块
            positions: List[Tuple[int, int]], 每个分块的 (y, x) 起始位置
            latent_shape: Tuple[int], 原始 latent 的形状 (B, C, H, W)
            tile_size: int, 每个分块的大小
            overlap_h: int, 垂直方向的重叠大小
            overlap_w: int, 水平方向的重叠大小
        Returns:
            latent: Tensor, 拼接后的 latent 表示
        """
        B, C, H, W = latent_shape
        latent = torch.zeros((B, C, H, W), device=tiles[0].device)
        weight = torch.zeros((B, C, H, W), device=tiles[0].device)  # 用于加权融合重叠区域

        stride_h = tile_size - overlap_h
        stride_w = tile_size - overlap_w

        for tile, (y, x) in zip(tiles, positions):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)

            latent[:, :, y:y_end, x:x_end] += tile[:, :, :y_end - y, :x_end - x]
            weight[:, :, y:y_end, x:x_end] += 1.0

        # 对重叠区域进行加权平均
        latent /= weight
        return latent



    def __call__(self,
                 prompt: str,
                 image_prompt: Image = None,
                 controlnet_image: Image = None,
                 width: int = 512,
                 height: int = 512,
                 guidance: float = 4,
                 num_steps: int = 50,
                 seed: int = 123456789,
                 true_gs: float = 3,
                 control_weight: float = 0.9,
                 ip_scale: float = 1.0,
                 neg_ip_scale: float = 1.0,
                 neg_prompt: str = '',
                 neg_image_prompt: Image = None,
                 timestep_to_start_cfg: int = 0,
                 ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)
        image_proj = None
        neg_image_proj = None
        if not (image_prompt is None and neg_image_prompt is None) :
            assert self.ip_loaded, 'You must setup IP-Adapter to add image prompt as input'

            if image_prompt is None:
                image_prompt = np.zeros((width, height, 3), dtype=np.uint8)
            if neg_image_prompt is None:
                neg_image_prompt = np.zeros((width, height, 3), dtype=np.uint8)

            image_proj = self.get_image_proj(image_prompt)
            neg_image_proj = self.get_image_proj(neg_image_prompt)

        if self.controlnet_loaded:
            # if width != height:
            #     raise ValueError(
            #         f"Controlnet generates only squared images, must have width=height, but get {width}x{height}"
            #     )
            # controlnet_image = self.annotator(controlnet_image, width, height)
            tile_vae = False
            # if not tile_vae:
            #     controlnet_image = controlnet_image.resize((1024, 1024))
            controlnet_image = torch.from_numpy((np.array(controlnet_image) / 127.5) - 1)
            controlnet_image = controlnet_image.permute(
                2, 0, 1).unsqueeze(0).to(torch.float).to(self.device)
            
            with torch.no_grad():
                controlnet_image1 = self.ae.encode(controlnet_image)
                controlnet_image1 = controlnet_image1.to(torch.bfloat16)
                # self.ae.decode(controlnet_image1)
                

            # print(controlnet_image1.shape)
            # assert(1==2)
                controlnet_image2 = self.ae2.encode(controlnet_image)
                controlnet_image2 = controlnet_image2.to(torch.bfloat16)
            
            if tile_vae:
                tile_size = 128
                overlap_h = 32 
                overlap_w = 32
                tiles1, positions1 = self.split_with_overlap(controlnet_image1, tile_size, overlap_h, overlap_w)
                tiles2, positions2 = self.split_with_overlap(controlnet_image2, tile_size, overlap_h, overlap_w)
                # print(tiles1.shape)
                print(len(tiles1), controlnet_image1.shape)
                processed_tiles = []
                for tile1, tile2 in zip(tiles1, tiles2):
                    print(tile1.shape)
                    with torch.no_grad():
                        # FLUX模型处理每个latent tile
                        controlnet_image = torch.cat([tile1, tile2], dim=0)

                        processed_tile = self.forward(
                                                    prompt,
                                                    width,
                                                    height,
                                                    guidance,
                                                    num_steps,
                                                    seed,
                                                    controlnet_image,
                                                    timestep_to_start_cfg=timestep_to_start_cfg,
                                                    true_gs=true_gs,
                                                    control_weight=control_weight,
                                                    neg_prompt=neg_prompt,
                                                    image_proj=image_proj,
                                                    neg_image_proj=neg_image_proj,
                                                    ip_scale=ip_scale,
                                                    neg_ip_scale=neg_ip_scale,
                                                    tile_vae=tile_vae
                                                )
                        processed_tile = unpack(processed_tile.float(), height, width)
                        processed_tiles.append(processed_tile)
                processed_latent = self.merge_with_overlap(processed_tiles, positions1, controlnet_image1.shape, tile_size, overlap_h, overlap_w)
                # x = unpack(processed_latent.float(), height, width)
                with torch.no_grad():
                    x = self.ae.decode(processed_latent)
                    self.offload_model_to_cpu(self.ae.decoder)

                x1 = x.clamp(-1, 1)
                x1 = rearrange(x1[-1], "c h w -> h w c")
                output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
                return output_img
            else:    

                # print(controlnet_image.shape)
                _, _, img_height, img_width = controlnet_image.shape
                controlnet_image = torch.cat([controlnet_image1, controlnet_image2], dim=0) # torch.Size([2, 16, h, w])
                
                # controlnet_image = transforms.ToTensor()(controlnet_image)
                # controlnet_image = F.interpolate(controlnet_image.unsqueeze(0), size=(1024, 1024), mode='bicubic')
                # controlnet_image = controlnet_image.to(torch.bfloat16).to(self.device)
                # img_lq = F.interpolate(img_lq.unsqueeze(0), size=(self.img_size, self.img_size), mode='bicubic')
                # print(controlnet_image.shape)

                return self.forward(
                    prompt,
                    width,
                    height,
                    guidance,
                    num_steps,
                    seed,
                    controlnet_image,
                    timestep_to_start_cfg=timestep_to_start_cfg,
                    true_gs=true_gs,
                    control_weight=control_weight,
                    neg_prompt=neg_prompt,
                    image_proj=image_proj,
                    neg_image_proj=neg_image_proj,
                    ip_scale=ip_scale,
                    neg_ip_scale=neg_ip_scale,
                    tile_vae=True,
                    img_height=img_height,
                    img_width=img_width
                )

    @torch.inference_mode()
    def gradio_generate(self, prompt, image_prompt, controlnet_image, width, height, guidance,
                        num_steps, seed, true_gs, ip_scale, neg_ip_scale, neg_prompt,
                        neg_image_prompt, timestep_to_start_cfg, control_type, control_weight,
                        lora_weight, local_path, lora_local_path, ip_local_path):
        if controlnet_image is not None:
            controlnet_image = Image.fromarray(controlnet_image)
            if ((self.controlnet_loaded and control_type != self.control_type)
                or not self.controlnet_loaded):
                if local_path is not None:
                    self.set_controlnet(control_type, local_path=local_path)
                else:
                    self.set_controlnet(control_type, local_path=None,
                                        repo_id=f"xlabs-ai/flux-controlnet-{control_type}-v3",
                                        name=f"flux-{control_type}-controlnet-v3.safetensors")
        if lora_local_path is not None:
            self.set_lora(local_path=lora_local_path, lora_weight=lora_weight)
        if image_prompt is not None:
            image_prompt = Image.fromarray(image_prompt)
            if neg_image_prompt is not None:
                neg_image_prompt = Image.fromarray(neg_image_prompt)
            if not self.ip_loaded:
                if ip_local_path is not None:
                    self.set_ip(local_path=ip_local_path)
                else:
                    self.set_ip(repo_id="xlabs-ai/flux-ip-adapter",
                                name="flux-ip-adapter.safetensors")
        seed = int(seed)
        if seed == -1:
            seed = torch.Generator(device="cpu").seed()

        img = self(prompt, image_prompt, controlnet_image, width, height, guidance,
                   num_steps, seed, true_gs, control_weight, ip_scale, neg_ip_scale, neg_prompt,
                   neg_image_prompt, timestep_to_start_cfg)

        filename = f"output/gradio/{uuid.uuid4()}.jpg"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Make] = "XLabs AI"
        exif_data[ExifTags.Base.Model] = self.model_type
        img.save(filename, format="jpeg", exif=exif_data, quality=95, subsampling=0)
        return img, filename


    def _gaussian_weights(self, tile_width, tile_height, in_channels, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, in_channels, 1, 1))

    def forward(
        self,
        prompt,
        width,
        height,
        guidance,
        num_steps,
        seed,
        controlnet_image = None,
        timestep_to_start_cfg = 0,
        true_gs = 3.5,
        control_weight = 0.9,
        neg_prompt="",
        image_proj=None,
        neg_image_proj=None,
        ip_scale=1.0,
        neg_ip_scale=1.0,
        tile_vae = False,
        img_height=1,
        img_width=1
    ):
        # print('width', height, width) #1024 1024
        if tile_vae:
            x = get_noise(
                1, height, width, device=self.device,
                dtype=torch.bfloat16, seed=seed
            )
        else:
            x = get_noise(
                1, height, width, device=self.device,
                dtype=torch.bfloat16, seed=seed
            )
        # print(x.shape) #torch.Size([1, 16, 128, 128]) latend size
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        # print(timesteps, num_steps)
        # [1.0, 0.9745572805404663, 0.9483310580253601, 0.921284556388855, 0.8933787941932678, 0.8645719885826111, 0.8348198533058167, 0.8040752410888672, 0.7722873091697693, 0.7394022941589355, 0.7053623795509338, 0.6701054573059082, 0.6335652470588684, 0.5956704616546631, 0.5563441514968872, 0.5155037045478821, 0.4730600416660309, 0.4289168119430542, 0.38296985626220703, 0.3351062536239624, 0.28520357608795166, 0.23312875628471375, 0.17873667180538177, 0.12186922132968903, 0.06235346570611, 0.0] 25
        torch.manual_seed(seed)
        
        latent_model_input = get_noise(
                1, img_height, img_width, device=self.device,
                dtype=torch.bfloat16, seed=seed
            )

        _, _, h, w = latent_model_input.size()
        tile_size, tile_overlap = (128, 8)
        if tile_vae:
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 16, 1) #torch.Size([1, 16, 128, 128])

            grid_rows = 0
            cur_x = 0
            while cur_x < latent_model_input.size(-1):
                cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < latent_model_input.size(-2):
                cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                grid_cols += 1

            input_list = []
            cond_list = []
            img_list = []
            noise_preds = []
            print('grid_rows, grid_cols', grid_rows, grid_cols)
            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    torch.cuda.empty_cache()
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    # input tile dimensions
                    print('latent_model_input', latent_model_input.shape)
                    print('controlnet_image', controlnet_image.shape)

                    input_tile = latent_model_input[:, :, input_start_y:input_end_y, input_start_x:input_end_x] # torch.Size([1, 16, 128, 128])
                    input_list.append(input_tile)
                    cond_tile = controlnet_image[:, :, input_start_y:input_end_y, input_start_x:input_end_x] # torch.Size([1, 16, 128, 128])
                    cond_list.append(cond_tile)
                    print('input_tile', input_tile.shape)
                    print('cond_tile', cond_tile.shape)
                    # img_tile = image[:, :, input_start_y*8:input_end_y*8, input_start_x*8:input_end_x*8]
                    # img_list.append(img_tile)
                    batch_size = 1
                    # if len(input_list) == batch_size or col == grid_cols-1:
                    if True:
    
                        # input_list_t = torch.cat(input_list, dim=0)
                        # cond_list_t = torch.cat(cond_list, dim=0)
                        # img_list_t = torch.cat(img_list, dim=0)
                        #print(input_list_t.shape, cond_list_t.shape, img_list_t.shape, fg_mask_list_t.shape)

                        # controlnet_image = torch.cat([input_tile, input_tile2], dim=0)

                        with torch.no_grad():
                            if self.offload:
                                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
                            inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
                            neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt)

                            if self.offload:
                                self.offload_model_to_cpu(self.t5, self.clip)
                                self.model = self.model.to(self.device)
                            if self.controlnet_loaded:
                                model_out = denoise_controlnet(
                                    self.model,
                                    **inp_cond,
                                    controlnet=self.controlnet,
                                    timesteps=timesteps,
                                    guidance=guidance,
                                    controlnet_cond=cond_tile,
                                    timestep_to_start_cfg=timestep_to_start_cfg,
                                    neg_txt=neg_inp_cond['txt'],
                                    neg_txt_ids=neg_inp_cond['txt_ids'],
                                    neg_vec=neg_inp_cond['vec'],
                                    true_gs=true_gs,
                                    controlnet_gs=control_weight,
                                    image_proj=image_proj,
                                    neg_image_proj=neg_image_proj,
                                    ip_scale=ip_scale,
                                    neg_ip_scale=neg_ip_scale,
                                )

                        #for sample_i in range(model_out.size(0)):
                        #    noise_preds_row.append(model_out[sample_i].unsqueeze(0))
                        # input_list = []
                        # cond_list = []
                        # img_list = []
                    print('model_out', model_out.shape)
                    model_out = unpack(model_out.float(), height, width)
                    print('model_out', model_out.shape)
                    noise_preds.append(model_out)

            # Stitch noise predictions for all tiles
            pred_ = torch.zeros(latent_model_input.shape, device=latent_model_input.device)
            contributors = torch.zeros(latent_model_input.shape, device=latent_model_input.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size
                    print(pred_.shape, len(noise_preds), noise_preds[0].shape)
                    pred_[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            # Average overlapping areas with more than 1 contributor
            pred_ /= contributors

            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(model_out.device)
            with torch.no_grad():
                torch.cuda.empty_cache()
                print('pred_', pred_.shape)
                x = self.ae.decode(pred_.float())
            self.offload_model_to_cpu(self.ae.decoder)
        
        else:
        
            with torch.no_grad():
                if self.offload:
                    self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
                inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
                neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt)

                if self.offload:
                    self.offload_model_to_cpu(self.t5, self.clip)
                    self.model = self.model.to(self.device)
                if self.controlnet_loaded:
                    x = denoise_controlnet(
                        self.model,
                        **inp_cond,
                        controlnet=self.controlnet,
                        timesteps=timesteps,
                        guidance=guidance,
                        controlnet_cond=controlnet_image,
                        timestep_to_start_cfg=timestep_to_start_cfg,
                        neg_txt=neg_inp_cond['txt'],
                        neg_txt_ids=neg_inp_cond['txt_ids'],
                        neg_vec=neg_inp_cond['vec'],
                        true_gs=true_gs,
                        controlnet_gs=control_weight,
                        image_proj=image_proj,
                        neg_image_proj=neg_image_proj,
                        ip_scale=ip_scale,
                        neg_ip_scale=neg_ip_scale,
                    )
                else:
                    x = denoise(
                        self.model,
                        **inp_cond,
                        timesteps=timesteps,
                        guidance=guidance,
                        timestep_to_start_cfg=timestep_to_start_cfg,
                        neg_txt=neg_inp_cond['txt'],
                        neg_txt_ids=neg_inp_cond['txt_ids'],
                        neg_vec=neg_inp_cond['vec'],
                        true_gs=true_gs,
                        image_proj=image_proj,
                        neg_image_proj=neg_image_proj,
                        ip_scale=ip_scale,
                        neg_ip_scale=neg_ip_scale,
                    )
                # if tile_vae:
                #     return x
                if self.offload:
                    self.offload_model_to_cpu(self.model)
                    self.ae.decoder.to(x.device)
                x = unpack(x.float(), height, width)
                x = self.ae.decode(x)
                self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        return output_img

    def offload_model_to_cpu(self, *models):
        if not self.offload: return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()


class XFluxPipeline_REIN:
    def __init__(self, model_type, device, offload: bool = False):
        self.device = torch.device(device)
        self.offload = offload
        self.model_type = model_type

        self.clip = load_clip(self.device)
        self.t5 = load_t5(self.device, max_length=512)
        self.ae = load_ae(model_type, device="cpu" if offload else self.device)
        
        self.ae2 = load_ae(model_type, device="cpu" if offload else self.device)
        self.ae2.encoder.load_state_dict(torch.load('experiments/uir_vae_encoder/checkpoint-2600/encoder_lq.bin'))

        if "fp8" in model_type:
            self.model = load_flow_model_quintized(model_type, device="cpu" if offload else self.device)
        else:
            self.model = load_flow_model(model_type, device="cpu" if offload else self.device)

        self.image_encoder_path = "openai/clip-vit-large-patch14"
        self.hf_lora_collection = "XLabs-AI/flux-lora-collection"
        self.lora_types_to_names = {
            "realism": "lora.safetensors",
        }
        self.controlnet_loaded = False
        self.ip_loaded = False

    def set_ip(self, local_path: str = None, repo_id = None, name: str = None):
        self.model.to(self.device)

        # unpack checkpoint
        checkpoint = load_checkpoint(local_path, repo_id, name)
        prefix = "double_blocks."
        blocks = {}
        proj = {}

        for key, value in checkpoint.items():
            if key.startswith(prefix):
                blocks[key[len(prefix):].replace('.processor.', '.')] = value
            if key.startswith("ip_adapter_proj_model"):
                proj[key[len("ip_adapter_proj_model."):]] = value

        for key, value in checkpoint.items():
            if key.startswith(prefix):
                blocks[key[len(prefix):].replace('.processor.', '.')] = value
            if key.startswith("ip_adapter_proj_model"):
                proj[key[len("ip_adapter_proj_model."):]] = value

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()

        # setup image embedding projection model
        self.improj = ImageProjModel(4096, 768, 4)
        self.improj.load_state_dict(proj)
        self.improj = self.improj.to(self.device, dtype=torch.bfloat16)

        ip_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            ip_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    ip_state_dict[k.replace(f'{name}.', '')] = checkpoint[k]
            if ip_state_dict:
                ip_attn_procs[name] = IPDoubleStreamBlockProcessor(4096, 3072)
                ip_attn_procs[name].load_state_dict(ip_state_dict)
                ip_attn_procs[name].to(self.device, dtype=torch.bfloat16)
            else:
                ip_attn_procs[name] = self.model.attn_processors[name]

        self.model.set_attn_processor(ip_attn_procs)
        self.ip_loaded = True

    def set_lora(self, local_path: str = None, repo_id: str = None,
                 name: str = None, lora_weight: int = 0.7):
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.update_model_with_lora(checkpoint, lora_weight)

    def set_lora_from_collection(self, lora_type: str = "realism", lora_weight: int = 0.7):
        checkpoint = load_checkpoint(
            None, self.hf_lora_collection, self.lora_types_to_names[lora_type]
        )
        self.update_model_with_lora(checkpoint, lora_weight)

    def update_model_with_lora(self, checkpoint, lora_weight):
        rank = get_lora_rank(checkpoint)
        lora_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            if name.startswith("single_blocks"):
                lora_attn_procs[name] = SingleStreamBlockProcessor()
                continue
            lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=rank)
            lora_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    lora_state_dict[k[len(name) + 1:]] = checkpoint[k] * lora_weight
            lora_attn_procs[name].load_state_dict(lora_state_dict)
            lora_attn_procs[name].to(self.device)

        self.model.set_attn_processor(lora_attn_procs)

    def set_controlnet(self, control_type: str, local_path: str = None, repo_id: str = None, name: str = None):
        self.model.to(self.device)
        self.controlnet = load_controlnet(self.model_type, self.device).to(torch.bfloat16)

        checkpoint = load_checkpoint(local_path, repo_id, name)
        print(local_path, repo_id, name)
        self.controlnet.load_state_dict(checkpoint, strict=False)
        self.annotator = Annotator(control_type, self.device)
        self.controlnet_loaded = True
        self.control_type = control_type

    def get_image_proj(
        self,
        image_prompt: Tensor,
    ):
        # encode image-prompt embeds
        image_prompt = self.clip_image_processor(
            images=image_prompt,
            return_tensors="pt"
        ).pixel_values
        image_prompt = image_prompt.to(self.image_encoder.device)
        image_prompt_embeds = self.image_encoder(
            image_prompt
        ).image_embeds.to(
            device=self.device, dtype=torch.bfloat16,
        )
        # encode image
        image_proj = self.improj(image_prompt_embeds)
        return image_proj

    def __call__(self,
                 prompt: str,
                 image_prompt: Image = None,
                 controlnet_image: Image = None,
                 width: int = 512,
                 height: int = 512,
                 guidance: float = 4,
                 num_steps: int = 50,
                 seed: int = 123456789,
                 true_gs: float = 3,
                 control_weight: float = 0.9,
                 ip_scale: float = 1.0,
                 neg_ip_scale: float = 1.0,
                 neg_prompt: str = '',
                 neg_image_prompt: Image = None,
                 timestep_to_start_cfg: int = 0,
                 ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)
        image_proj = None
        neg_image_proj = None
        if not (image_prompt is None and neg_image_prompt is None) :
            assert self.ip_loaded, 'You must setup IP-Adapter to add image prompt as input'

            if image_prompt is None:
                image_prompt = np.zeros((width, height, 3), dtype=np.uint8)
            if neg_image_prompt is None:
                neg_image_prompt = np.zeros((width, height, 3), dtype=np.uint8)

            image_proj = self.get_image_proj(image_prompt)
            neg_image_proj = self.get_image_proj(neg_image_prompt)

        if self.controlnet_loaded:
            if width != height:
                raise ValueError(
                    f"Controlnet generates only squared images, must have width=height, but get {width}x{height}"
                )
            # controlnet_image = self.annotator(controlnet_image, width, height)
            controlnet_image = controlnet_image.resize((1024, 1024))
            controlnet_image = torch.from_numpy((np.array(controlnet_image) / 127.5) - 1)
            controlnet_image = controlnet_image.permute(
                2, 0, 1).unsqueeze(0).to(torch.float).to(self.device)
            
            controlnet_image1 = self.ae.encode(controlnet_image)
            controlnet_image1 = controlnet_image1.to(torch.bfloat16)

            controlnet_image2 = self.ae2.encode(controlnet_image)
            controlnet_image2 = controlnet_image2.to(torch.bfloat16)

            controlnet_image = torch.cat([controlnet_image1, controlnet_image2], dim=0)
            
            # controlnet_image = transforms.ToTensor()(controlnet_image)
            # controlnet_image = F.interpolate(controlnet_image.unsqueeze(0), size=(1024, 1024), mode='bicubic')
            # controlnet_image = controlnet_image.to(torch.bfloat16).to(self.device)
            # img_lq = F.interpolate(img_lq.unsqueeze(0), size=(self.img_size, self.img_size), mode='bicubic')
            # print(controlnet_image.shape)

        return self.forward(
            prompt,
            width,
            height,
            guidance,
            num_steps,
            seed,
            controlnet_image,
            timestep_to_start_cfg=timestep_to_start_cfg,
            true_gs=true_gs,
            control_weight=control_weight,
            neg_prompt=neg_prompt,
            image_proj=image_proj,
            neg_image_proj=neg_image_proj,
            ip_scale=ip_scale,
            neg_ip_scale=neg_ip_scale,
        )

    @torch.inference_mode()
    def gradio_generate(self, prompt, image_prompt, controlnet_image, width, height, guidance,
                        num_steps, seed, true_gs, ip_scale, neg_ip_scale, neg_prompt,
                        neg_image_prompt, timestep_to_start_cfg, control_type, control_weight,
                        lora_weight, local_path, lora_local_path, ip_local_path):
        if controlnet_image is not None:
            controlnet_image = Image.fromarray(controlnet_image)
            if ((self.controlnet_loaded and control_type != self.control_type)
                or not self.controlnet_loaded):
                if local_path is not None:
                    self.set_controlnet(control_type, local_path=local_path)
                else:
                    self.set_controlnet(control_type, local_path=None,
                                        repo_id=f"xlabs-ai/flux-controlnet-{control_type}-v3",
                                        name=f"flux-{control_type}-controlnet-v3.safetensors")
        if lora_local_path is not None:
            self.set_lora(local_path=lora_local_path, lora_weight=lora_weight)
        if image_prompt is not None:
            image_prompt = Image.fromarray(image_prompt)
            if neg_image_prompt is not None:
                neg_image_prompt = Image.fromarray(neg_image_prompt)
            if not self.ip_loaded:
                if ip_local_path is not None:
                    self.set_ip(local_path=ip_local_path)
                else:
                    self.set_ip(repo_id="xlabs-ai/flux-ip-adapter",
                                name="flux-ip-adapter.safetensors")
        seed = int(seed)
        if seed == -1:
            seed = torch.Generator(device="cpu").seed()

        img = self(prompt, image_prompt, controlnet_image, width, height, guidance,
                   num_steps, seed, true_gs, control_weight, ip_scale, neg_ip_scale, neg_prompt,
                   neg_image_prompt, timestep_to_start_cfg)

        filename = f"output/gradio/{uuid.uuid4()}.jpg"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Make] = "XLabs AI"
        exif_data[ExifTags.Base.Model] = self.model_type
        img.save(filename, format="jpeg", exif=exif_data, quality=95, subsampling=0)
        return img, filename

    def forward(
        self,
        prompt,
        width,
        height,
        guidance,
        num_steps,
        seed,
        controlnet_image = None,
        timestep_to_start_cfg = 0,
        true_gs = 3.5,
        control_weight = 0.9,
        neg_prompt="",
        image_proj=None,
        neg_image_proj=None,
        ip_scale=1.0,
        neg_ip_scale=1.0,
    ):
        x = get_noise(
            1, height, width, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )
        # print(x.shape) #torch.Size([1, 16, 128, 128]) latend size
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        # print(timesteps, num_steps)
        # [1.0, 0.9745572805404663, 0.9483310580253601, 0.921284556388855, 0.8933787941932678, 0.8645719885826111, 0.8348198533058167, 0.8040752410888672, 0.7722873091697693, 0.7394022941589355, 0.7053623795509338, 0.6701054573059082, 0.6335652470588684, 0.5956704616546631, 0.5563441514968872, 0.5155037045478821, 0.4730600416660309, 0.4289168119430542, 0.38296985626220703, 0.3351062536239624, 0.28520357608795166, 0.23312875628471375, 0.17873667180538177, 0.12186922132968903, 0.06235346570611, 0.0] 25
        torch.manual_seed(seed)
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
            neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt)

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)
            if self.controlnet_loaded:
                X_t = denoise_controlnet_rein(
                    self.model,
                    **inp_cond,
                    controlnet=self.controlnet,
                    timesteps=timesteps,
                    guidance=guidance,
                    controlnet_cond=controlnet_image,
                    timestep_to_start_cfg=timestep_to_start_cfg,
                    neg_txt=neg_inp_cond['txt'],
                    neg_txt_ids=neg_inp_cond['txt_ids'],
                    neg_vec=neg_inp_cond['vec'],
                    true_gs=true_gs,
                    controlnet_gs=control_weight,
                    image_proj=image_proj,
                    neg_image_proj=neg_image_proj,
                    ip_scale=ip_scale,
                    neg_ip_scale=neg_ip_scale,
                )
            else:
                x = denoise(
                    self.model,
                    **inp_cond,
                    timesteps=timesteps,
                    guidance=guidance,
                    timestep_to_start_cfg=timestep_to_start_cfg,
                    neg_txt=neg_inp_cond['txt'],
                    neg_txt_ids=neg_inp_cond['txt_ids'],
                    neg_vec=neg_inp_cond['vec'],
                    true_gs=true_gs,
                    image_proj=image_proj,
                    neg_image_proj=neg_image_proj,
                    ip_scale=ip_scale,
                    neg_ip_scale=neg_ip_scale,
                )

            x = X_t[-1]
            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(x.device)
            x = unpack(x.float(), height, width)
            x = self.ae.decode(x)
            self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        return output_img, X_t

    def offload_model_to_cpu(self, *models):
        if not self.offload: return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()

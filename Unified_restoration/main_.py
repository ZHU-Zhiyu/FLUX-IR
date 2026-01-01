import argparse
from PIL import Image
import os

from src.flux.xflux_pipeline import XFluxPipeline
from ram.models.ram_lora import ram
from ram import inference_ram as inference
from torchvision import transforms
import numpy as np


from PIL import Image
import torchvision.transforms.functional as F
import torch

# Define the patch size and overlap
patch_size = (1536, 1180)
overlap = 200  # Amount of overlap between patches

# ... (split_image and combine_patches functions remain the same) ...

# Modified split_image function to handle overlap
def split_image(image, patch_size, overlap):
    width, height = image.size
    patches = []
    for i in range(0, height - overlap, patch_size[1] - overlap):
        for j in range(0, width - overlap, patch_size[0] - overlap):
            box = (j, i, j + patch_size[0], i + patch_size[1])
            patch = image.crop(box)
            patches.append(patch)
    return patches

def create_weight_mask(patch_size, overlap):
    # patch_size应该是(height, width)的元组
    height, width = patch_size
    mask = np.ones((height, width))
    
    # 水平方向的渐变
    for i in range(overlap):
        mask[:, i] *= i / overlap  # 左边缘
        mask[:, -(i+1)] *= i / overlap  # 右边缘
    # 垂直方向的渐变
    for i in range(overlap):
        mask[i, :] *= i / overlap  # 上边缘
        mask[-(i+1), :] *= i / overlap  # 下边缘
    return mask

def combine_patches(patches, image_size, patch_size, overlap):
    width, height = image_size
    new_image = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    
    # 创建权重掩码，注意patch_size的顺序
    weight_mask = create_weight_mask((patch_size[1], patch_size[0]), overlap)
    
    index = 0
    for i in range(0, height - overlap, patch_size[1] - overlap):
        for j in range(0, width - overlap, patch_size[0] - overlap):
            max_height = min(i + patch_size[1], height)
            max_width = min(j + patch_size[0], width)
            
            patch_np = np.array(patches[index], dtype=np.float32)
            h, w = max_height-i, max_width-j
            
            # 应用权重掩码，注意广播
            patch_weight = weight_mask[:h, :w, np.newaxis]
            weighted_patch = patch_np[:h, :w] * patch_weight
            
            new_image[i:max_height, j:max_width] += weighted_patch
            weight_sum[i:max_height, j:max_width] += weight_mask[:h, :w]
            
            index += 1
    
    # 归一化，注意广播
    weight_sum = weight_sum[..., np.newaxis]
    new_image = new_image / (weight_sum + 1e-6)
    return Image.fromarray(np.uint8(new_image))

tensor_transforms = transforms.Compose([transforms.ToTensor(),])
ram_transforms = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ])

ram_model = ram(pretrained='ram_model/ram_swin_large_14m.pth',
            pretrained_condition='ram_model/DAPE.pth',
            image_size=384,
            vit='swin_l')
ram_model.eval()
ram_model.cuda()

def get_prompt(image, model):
    validation_prompt = ""
 
    image = tensor_transforms(image).unsqueeze(0).cuda()
    image = ram_transforms(image)
    res = inference(image, model)
    ram_encoder_hidden_states = model.generate_image_embeds(image)

    # validation_prompt = f"{res[0]},"
    validation_prompt = res[0]

    return validation_prompt, ram_encoder_hidden_states

def create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", type=str, required=True,
        help="The input text prompt"
    )
    parser.add_argument(
        "--neg_prompt", type=str, default="",
        help="The input text negative prompt"
    )
    parser.add_argument(
        "--img_prompt", type=str, default=None,
        help="Path to input image prompt"
    )
    parser.add_argument(
        "--neg_img_prompt", type=str, default=None,
        help="Path to input negative image prompt"
    )
    parser.add_argument(
        "--ip_scale", type=float, default=1.0,
        help="Strength of input image prompt"
    )
    parser.add_argument(
        "--neg_ip_scale", type=float, default=1.0,
        help="Strength of negative input image prompt"
    )
    parser.add_argument(
        "--local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (Controlnet)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="A filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (IP-Adapter)"
    )
    parser.add_argument(
        "--ip_name", type=str, default=None,
        help="A IP-Adapter filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_local_path", type=str, default=None,
        help="Local path to the model checkpoint (IP-Adapter)"
    )
    parser.add_argument(
        "--lora_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (LoRA)"
    )
    parser.add_argument(
        "--lora_name", type=str, default=None,
        help="A LoRA filename to download from HuggingFace"
    )
    parser.add_argument(
        "--lora_local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--offload", action='store_true', help="Offload model to CPU when not in use"
    )
    parser.add_argument(
        "--use_ip", action='store_true', help="Load IP model"
    )
    parser.add_argument(
        "--use_lora", action='store_true', help="Load Lora model"
    )
    parser.add_argument(
        "--use_controlnet", action='store_true', help="Load Controlnet model"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1,
        help="The number of images to generate per prompt"
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to image"
    )
    parser.add_argument(
        "--lora_weight", type=float, default=0.9, help="Lora model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--control_weight", type=float, default=0.8, help="Controlnet model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--control_type", type=str, default="canny",
        choices=("canny", "openpose", "depth", "hed", "hough", "tile"),
        help="Name of controlnet condition, example: canny"
    )
    parser.add_argument(
        "--model_type", type=str, default="flux-dev",
        choices=("flux-dev", "flux-dev-fp8", "flux-schnell"),
        help="Model type to use (flux-dev, flux-dev-fp8, flux-schnell)"
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="The width for generated image"
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="The height for generated image"
    )
    parser.add_argument(
        "--num_steps", type=int, default=25, help="The num_steps for diffusion process"
    )
    parser.add_argument(
        "--guidance", type=float, default=4, help="The guidance for diffusion process"
    )
    parser.add_argument(
        "--seed", type=int, default=123456789, help="A seed for reproducible inference"
    )
    parser.add_argument(
        "--true_gs", type=float, default=3.5, help="true guidance"
    )
    parser.add_argument(
        "--timestep_to_start_cfg", type=int, default=5, help="timestep to start true guidance"
    )
    parser.add_argument(
        "--save_path", type=str, default='results', help="Path to save"
    )
    parser.add_argument(
        "--images_path", type=str, default='results', help="Images to test"
    )
    return parser


def main(args):
    if args.image:
        image = Image.open(args.image).convert('RGB')
    else:
        image = None

    xflux_pipeline = XFluxPipeline(args.model_type, args.device, args.offload)
    if args.use_ip:
        print('load ip-adapter:', args.ip_local_path, args.ip_repo_id, args.ip_name)
        xflux_pipeline.set_ip(args.ip_local_path, args.ip_repo_id, args.ip_name)
    if args.use_lora:
        print('load lora:', args.lora_local_path, args.lora_repo_id, args.lora_name)
        xflux_pipeline.set_lora(args.lora_local_path, args.lora_repo_id, args.lora_name, args.lora_weight)
    if args.use_controlnet:
        print('load controlnet:', args.local_path, args.repo_id, args.name)
        xflux_pipeline.set_controlnet(args.control_type, args.local_path, args.repo_id, args.name)

    image_prompt = Image.open(args.img_prompt) if args.img_prompt else None
    neg_image_prompt = Image.open(args.neg_img_prompt) if args.neg_img_prompt else None


    val_images = [os.path.join(args.images_path, i) \
            for i in os.listdir(args.images_path) if '.jpg' in i or '.png' in i]
    val_images.sort()
    # val_images = val_images[0:25]
    seed = np.random.randint(1, 10000)
    seed = 1234
    print(seed) #3485 9532 1234
    for i in range(len(val_images)):
        
        val_image_path = val_images[i]
        image_name = val_image_path.split('.')[0].split('low/')[1]
        image = Image.open(val_image_path).convert('RGB')

        print(f'--------------------- processing number No.{i} image -{image_name}  ----------------------')
        print(image.size)
        # ram_image = tensor_transforms(image)
        # ram_image = ram_transforms(ram_image)
        # print(ram_image.shape)
        ram_prompt, _ = \
            get_prompt(image, ram_model)

        prompts = f"{ram_prompt}, {args.prompt}"
        print(prompts)
        # prompts = ''
        # seed = 1234 #7018 1458
        # seed = 1458


        # Split the image into patches
        patches = split_image(image, patch_size, overlap)

        # Placeholder for processed patches
        processed_patches = []

        # Assume `model` is your neural network with encoder and decoder
        # model = YourModel()
        torch.cuda.synchronize()
        import time
        start_time = time.time() 
        i = 0
        for patch in patches:
            i = i + 1
            print(f'patch {i}/{len(patches)}')
            for _ in range(args.num_images_per_prompt):
                processed_patch = xflux_pipeline(
                    prompt=prompts,
                    controlnet_image=patch,
                    width=args.width,
                    height=args.height,
                    guidance=args.guidance,
                    num_steps=args.num_steps,
                    seed=seed,
                    true_gs=args.true_gs,
                    control_weight=args.control_weight,
                    neg_prompt=args.neg_prompt,
                    timestep_to_start_cfg=args.timestep_to_start_cfg,
                    image_prompt=image_prompt,
                    neg_image_prompt=neg_image_prompt,
                    ip_scale=args.ip_scale,
                    neg_ip_scale=args.neg_ip_scale,
                )
                processed_patch = processed_patch.resize(patch_size)
                processed_patches.append(processed_patch)

        # Combine processed patches into a full image
        result = combine_patches(processed_patches, image.size, patch_size, overlap)

        torch.cuda.synchronize()
        end_time=time.time() 
        #print(end_time)
        onceTime = end_time-start_time
        print('onceTime: {}s'.format(onceTime))
        # print(result.size)
        # result = result.resize(image.size)
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        ind = len(os.listdir(args.save_path))
        result.save(os.path.join(args.save_path, f"{image_name}.png"))
        args.seed = args.seed + 1


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)

# Unified Image Restoration

For unified image restoration, we trained a single diffusion model based on FLUX-dev to handle various types of degradation, including super-resolution, denoising, low-light enhancement, etc.

## Quick Start

### Dependencies and Installation

- Python 3.11
- Pytorch >= 2.4
- CUDA >= 12
- HuggingFace CLI

1. Create Conda Environment

```
conda create --name FluxIR python=3.11
conda activate FluxIR 
```

2. Clone Repo

```
git clone https://github.com/ZHU-Zhiyu/FLUX-IR.git
```

3. Install Dependencies

```
cd FLUX-IR/Unified_restoration/
pip install -r requirements.txt
```

### Testing

You can refer to the following links to download the [pretrained model](https://drive.google.com/drive/folders/1CFWxmxOwcp6ARRX-y9yYsXSwpIRAgK37?usp=sharing) and put it in the following folder:

```
├── checkpoints
    ├── FluxIR.bin
    ├── encoder_lq.bin
```

```
# Super-resolution
CUDA_VISIBLE_DEVICES="0" python main.py \
 --prompt 'high-resolution, ultra-sharp, detailed' \
 --images_path input/super-resolution \
 --local_path checkpoints/FluxIR.bin \
 --use_controlnet \
 --model_type flux-dev \
 --width 1024 --height 1024  --timestep_to_start_cfg 5 \
 --num_steps 21 --true_gs 4 --guidance 4 \
 --control_weight 0.8 \
 --save_path results/super-resolution
```

```
# Image denoising
CUDA_VISIBLE_DEVICES="0" python main.py \
 --prompt 'noise-free, clean, smooth' \
 --images_path input/noisy \
 --local_path checkpoints/FluxIR.bin \
 --use_controlnet \
 --model_type flux-dev \
 --width 1024 --height 1024  --timestep_to_start_cfg 5 \
 --num_steps 21 --true_gs 4 --guidance 4 \
 --control_weight 0.8 \
 --save_path results/denoising
```

```
# Low-light enhancement 
CUDA_VISIBLE_DEVICES="0" python main.py \
 --prompt 'bright, clear, vivid' \
 --images_path input/llie \
 --local_path checkpoints/FluxIR.bin \
 --use_controlnet \
 --model_type flux-dev \
 --width 1024 --height 1024  --timestep_to_start_cfg 5 \
 --num_steps 21 --true_gs 4 --guidance 4 \
 --control_weight 0.9 \
 --save_path results/llie
```

```
# Raindrop removal
CUDA_VISIBLE_DEVICES="0" python main.py \
 --prompt 'remove raindrops, clean' \
 --images_path input/raindrop \
 --local_path checkpoints/FluxIR.bin \
 --use_controlnet \
 --model_type flux-dev \
 --width 1024 --height 1024  --timestep_to_start_cfg 5 \
 --num_steps 21 --true_gs 4 --guidance 4 \
 --control_weight 0.9 \
 --save_path results/raindrop
```

## Acknowledgement

Our code is built upon [X-FLUX](https://github.com/XLabs-AI/x-flux). Thanks to the contributors for their great work.

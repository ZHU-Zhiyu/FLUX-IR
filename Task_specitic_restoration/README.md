# Task-specitic restoration

For task-specific image restoration, e.g., low-light enhancement, raindrop removal, and deraining, we trained a diffusion model for each particular degradation.

## Quick Start

### Dependencies and Installation

- Python 3.8
- Pytorch 1.11

1. Create Conda Environment

```
conda create --name EETDiff python=3.8
conda activate EETDiff 
```

2. Install PyTorch

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

3. Clone Repo

```
git clone https://github.com/jinnh/GSAD.git](https://github.com/ZHU-Zhiyu/FLUX-IR.git
```

4. Install Dependencies

```
cd FLUX-IR/Task_specitic_restoration/
pip install -r requirements.txt
```

### Testing

You can refer to the following links to download the [pretrained model](https://drive.google.com/drive/folders/1CFWxmxOwcp6ARRX-y9yYsXSwpIRAgK37?usp=sharing) and put it in the following folder:

```
├── checkpoints
  ├── lolv1
    	├── lolv1_dist1step.pth
    	├── lolv1_dist2step.pth
    	├── lolv1_rein.pth
  ├── raindrop
    	├── raindrop_dist1step.pth
    	├── raindrop_dist2step.pth
   	├── raindrop_rein.pth
```

```
# LOLv1
// Testing Reinforcing ODE Trajectories with Modulated SDEs
python test.py --dataset ./config/lolv1.yml --config ./config/lolv1_test.json --stage rein

// Testing Cost-aware trajectory distillation
python test.py --dataset ./config/lolv1.yml --config ./config/lolv1_test.json --stage dist

# raindrop
// Testing Reinforcing ODE Trajectories with Modulated SDEs
python test.py --config config/raindrop_test.json --dataset ./config/raindrop.yml --stage rein

// Testing Cost-aware trajectory distillation
python test.py --config config/raindrop_test.json --dataset ./config/raindrop.yml --stage dist
```

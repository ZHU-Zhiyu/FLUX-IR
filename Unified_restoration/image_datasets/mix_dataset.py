import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import cv2
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional
from .realesrgan import RealESRGAN_degradation
from torchvision.transforms import ToPILImage

tensor_transforms = transforms.Compose([transforms.ToTensor(),])
ram_transforms = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ])

def random_crop(image1, image2, crop_size):
    image_width, image_height = image1.size
    crop_height, crop_width = crop_size
    assert image_width >= crop_width and image_height >= crop_height, "裁剪尺寸不能大于图像尺寸。"
    x = random.randint(0, image_width - crop_width)
    y = random.randint(0, image_height - crop_height)
    crop1 = image1.crop((x, y, x + crop_width, y + crop_height))
    crop2 = image2.crop((x, y, x + crop_width, y + crop_height))
    return crop1, crop2

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512):
        
        self.deg_types =['super-resolution', 'super-resolution', 'super-resolution',  \
                         'motion-blurry', 'motion-blurry', 'noisy', 'noisy', 'outdoor-rain', 'super-resolution', \
                         'underwater', 'super-resolution', 'snowy', 'outdoor-rain', \
                         'raindrop', 'low-light',]
        self.distortion = {}
        self.data_len = 0
        self.train_img = []
        for deg_type in self.deg_types:
            
            images_gt = [os.path.join(img_dir+'/'+deg_type+'/high', i) \
                for i in os.listdir(img_dir+'/'+deg_type+'/high') if '.jpg' in i or '.png' in i]
            images_gt.sort()

            self.data_len = self.data_len + len(images_gt)
            self.distortion[deg_type] = images_gt
        self.data_lens = [len(self.distortion[deg_type]) for deg_type in self.deg_types]

        self.img_size = img_size
        self.degradation = RealESRGAN_degradation('image_datasets/params_realesrgan.yml', device='cpu')
        
        
        self.img_preproc = transforms.Compose([       
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        try:
            
            type_id = int(idx % len(self.deg_types))
            deg_type = self.deg_types[type_id]
            idx = np.random.randint(self.data_lens[type_id])
            

            GT_path = self.distortion[deg_type][idx]
            img_gt = Image.open(GT_path)
            
            # get LQ image
            LQ_path = GT_path.replace('high', 'low')
            if deg_type == 'noisy' or deg_type == 'super-resolution' or deg_type == 'demoire':

                img_lq = Image.open(LQ_path)
                
                if img_gt.size[0] > 1024 and img_gt.size[1] > 1024:
                    img_gt, img_lq = random_crop(img_gt, img_lq, (self.img_size, self.img_size), overlap_ratio=0.1)
                else:
                    img_gt = img_gt.resize((self.img_size, self.img_size))
                    img_lq = img_lq.resize((self.img_size, self.img_size))

                raw_size = img_gt.size

                ram_image = tensor_transforms(img_lq)
                ram_image = ram_transforms(ram_image)                

                img_gt = torch.from_numpy((np.array(img_gt) / 127.5) - 1)
                img_gt = img_gt.permute(2, 0, 1)

                img_lq = torch.from_numpy((np.array(img_lq) / 127.5) - 1)
                img_lq = img_lq.permute(2, 0, 1)

            else: 
                raw_size = img_gt.size
                if deg_type == 'outdoor-rain':
                    img_name = LQ_path.split('im_')[1]
                    img_name = 'im_'+ img_name.split('.')[0] #im_0001
                    
                    rain_type = [
                        '_s100_a04', '_s100_a05', '_s100_a06', \
                        '_s95_a04', '_s95_a05', '_s95_a06', \
                        '_s90_a04', '_s90_a05', '_s90_a06', \
                        '_s85_a04', '_s85_a05', '_s85_a06', \
                        '_s80_a04', '_s80_a05', '_s80_a06', \
                        ]
                    
                    rain_type_id = np.random.randint(len(rain_type))
                    LQ_path = LQ_path.replace(img_name, img_name + rain_type[rain_type_id])
                elif deg_type == 'exposure_error':
                    
                    img_name = GT_path.split('high/')[1].split('.')[0]
                    exposure_type = ['_0.JPG', '_N1.5.JPG', '_N1.JPG', '_P1.5.JPG', '_P1.JPG']
                    exposure_type_id = np.random.randint(len(exposure_type))
                    LQ_path = LQ_path.replace(img_name+'.jpg', img_name + exposure_type[exposure_type_id])

                elif deg_type == 'raindrop':
                    LQ_path = LQ_path.replace('clean', 'rain')
                img_lq = Image.open(LQ_path)
                
                ram_image = tensor_transforms(img_lq)
                ram_image = ram_transforms(ram_image)

                img_gt = img_gt.resize((self.img_size, self.img_size))
                img_gt = torch.from_numpy((np.array(img_gt) / 127.5) - 1)
                img_gt = img_gt.permute(2, 0, 1)

                img_lq = img_lq.resize((self.img_size, self.img_size))
                img_lq = torch.from_numpy((np.array(img_lq) / 127.5) - 1)
                img_lq = img_lq.permute(2, 0, 1)

            print(deg_type, GT_path)
            if deg_type == 'super-resolution':
                prompt = 'high-resolution, ultra-sharp, detailed'
            elif deg_type == 'motion-blurry':
                prompt = 'sharp, deblurred, clear'
            elif deg_type == 'low-light':
                prompt = 'bright, clear, vivid'                
            elif deg_type == 'noisy':
                prompt = 'noise-free, clean, smooth'   
            elif deg_type == 'raindrop':
                prompt = 'remove raindrops, clean'
            elif deg_type == 'outdoor-rain':
                prompt = 'remove rain streaks, dehaze, improve visibility'
            elif deg_type == 'snowy':
                prompt = 'remove snowflakes, snow-free'
            elif deg_type == 'underwater':
                prompt = 'clear, reduce backscatter, restore colors'

            return img_gt, img_lq, prompt, ram_image, raw_size, GT_path
        except Exception as e:
            print(e)
            print(deg_type)
            print(LQ_path)

            return self.__getitem__(random.randint(0, len(self.images) - 1))


def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)

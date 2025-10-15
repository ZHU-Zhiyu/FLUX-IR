import os
import torch.utils.data as data
import numpy as np
import torch
import cv2
from torchvision.transforms import ToTensor
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
import random
import natsort

class DID_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("patch_size", None)
        if train:
            self.split = 'train'
            self.root = os.path.join(self.root, 'train')
        else:
            self.split = 'val'
            self.root = os.path.join(self.root, 'test')
        self.pairs = self.load_pairs(self.root)
        self.to_tensor = ToTensor()
        
        self.small_rain = self.pairs[:6000]  # 小雨条图像
        self.large_rain = self.pairs[6000:] 

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):

        # low_list = os.listdir(os.path.join(folder_path, 'low'))
        low_list = natsort.natsorted(os.listdir(os.path.join(folder_path, 'low')),alg = natsort.ns.PATH)

        # low_list.sort(key=lambda x:int(x.split('.')[0]))
        low_list = filter(lambda x: 'jpg' or 'png' in x, low_list)
        
        
        pairs = []
        for idx, f_name in enumerate(low_list):
            # print(idx, f_name)
            if self.split == 'val':
                # print(f_name, f_name[0:len(f_name)-9]+'_clean.png')
                pairs.append(
                    [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB),
                    #  cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name.split('_')[0]+'.jpg')), cv2.COLOR_BGR2RGB), 
                     cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name)), cv2.COLOR_BGR2RGB), 
                    #  cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name.split('x')[0]+'.png')), cv2.COLOR_BGR2RGB),
                    f_name.split('.')[0]])
            else:
                # print(f_name)
                pairs.append(
                    [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB), 
                    #  cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name.split('_')[0]+'.jpg')), cv2.COLOR_BGR2RGB), 
                     cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name)), cv2.COLOR_BGR2RGB), 
                    #  cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name.split('x')[0]+'.png')), cv2.COLOR_BGR2RGB),
                    f_name.split('.')[0]])
        return pairs

    def __getitem__(self, item):
        
        if self.split == 'val':
            lr, hr, f_name = self.pairs[item]
        else:
            if item % 2 == 0:
                lr, hr, f_name = self.small_rain[item // 2]
            else:
                lr, hr, f_name = self.large_rain[5999-item // 2]
            
        # lr, hr, f_name = self.pairs[item]
        # print('item', item, index, f_name)
        # hr = hr[0:480, 0:720, :]
        # lr = lr[0:480, 0:720, :]
        # print(f_name, hr.shape, lr.shape, self.crop_size)


        if self.use_crop and self.split != 'val':
            hr, lr = random_crop(hr, lr, self.crop_size)
        elif self.split == 'val':
            # print(hr.shape)
            # lr = cv2.resize(lr, (480, 480))
            # hr = cv2.resize(hr, (480, 480))
            lr = cv2.resize(lr, (lr.shape[1] // 16 * 16, lr.shape[0] // 16 * 16))
            hr = cv2.resize(hr, (hr.shape[1] // 16 * 16, hr.shape[0] // 16 * 16))
            # print(hr.shape)

        # elif self.use_crop and self.split == 'val':
        #     h = int(hr.shape[0]/32)
        #     w = int(hr.shape[1]/32)
        #     # print(hr.shape)
        #     # print(h,w)
        #     # print(int((hr.shape[0]-h*32)/2), int(hr.shape[0]-(hr.shape[0]-h*32)/2))
        #     # assert(1==2)
        #     hr = hr[int((hr.shape[0]-h*32)/2):int(hr.shape[0]-(hr.shape[0]-h*32)/2), int((hr.shape[1]-w*32)/2):int(hr.shape[1]-(hr.shape[1]-w*32)/2), :]
        #     lr = lr[int((hr.shape[0]-h*32)/2):int(hr.shape[0]-(hr.shape[0]-h*32)/2), int((hr.shape[1]-w*32)/2):int(hr.shape[1]-(hr.shape[1]-w*32)/2), :]

        # if self.center_crop_hr_size:
        #     hr, lr = center_crop(hr, self.center_crop_hr_size), center_crop(lr, self.center_crop_hr_size)

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)


        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        # print(f_name, hr.shape, lr.shape)

        [lr, hr] = transform_augment(
                [lr, hr], split=self.split, min_max=(-1, 1))
        # print(f_name)
        return {'LQ': lr, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name}

class DDN_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("patch_size", None)
        if train:
            self.split = 'train'
            self.root = os.path.join(self.root, 'train')
        else:
            self.split = 'val'
            self.root = os.path.join(self.root, 'test8')
        self.pairs = self.load_pairs(self.root)
        self.to_tensor = ToTensor()
        

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):

        low_list = os.listdir(os.path.join(folder_path, 'low'))
        # low_list = natsort.natsorted(os.listdir(os.path.join(folder_path, 'low')),alg = natsort.ns.PATH)

        # low_list.sort(key=lambda x:int(x.split('.')[0]))
        low_list = filter(lambda x: 'jpg' or 'png' in x, low_list)
        
        
        pairs = []
        for idx, f_name in enumerate(low_list):
            # print(idx, f_name)
            if self.split == 'val':
                # print(f_name, f_name[0:len(f_name)-9]+'_clean.png')
                pairs.append(
                    [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB),
                     cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name.split('_')[0]+'.jpg')), cv2.COLOR_BGR2RGB), 
                    #  cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name)), cv2.COLOR_BGR2RGB), 
                    #  cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name.split('x')[0]+'.png')), cv2.COLOR_BGR2RGB),
                    f_name.split('.')[0]])
            else:
                # print(f_name)
                pairs.append(
                    [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB), 
                     cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name.split('_')[0]+'.jpg')), cv2.COLOR_BGR2RGB), 
                    #  cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name)), cv2.COLOR_BGR2RGB), 
                    #  cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name.split('x')[0]+'.png')), cv2.COLOR_BGR2RGB),
                    f_name.split('.')[0]])
        return pairs

    def __getitem__(self, item):
            
        lr, hr, f_name = self.pairs[item]
        # print('item', item, index, f_name)
        # hr = hr[0:480, 0:720, :]
        # lr = lr[0:480, 0:720, :]
        # print(f_name, hr.shape, lr.shape, self.crop_size)


        if self.use_crop and self.split != 'val':
            if '837' not in f_name:
                hr, lr = random_crop(hr, lr, self.crop_size)
            else:
                lr = cv2.resize(lr, (self.crop_size, self.crop_size))
                hr = cv2.resize(hr, (self.crop_size, self.crop_size))   
        elif self.split == 'val':
            # print(hr.shape)
            # lr = cv2.resize(lr, (480, 480))
            # hr = cv2.resize(hr, (480, 480))
            lr = cv2.resize(lr, (lr.shape[1] // 16 * 16, lr.shape[0] // 16 * 16))
            hr = cv2.resize(hr, (hr.shape[1] // 16 * 16, hr.shape[0] // 16 * 16))
            # print(hr.shape)

        # elif self.use_crop and self.split == 'val':
        #     h = int(hr.shape[0]/32)
        #     w = int(hr.shape[1]/32)
        #     # print(hr.shape)
        #     # print(h,w)
        #     # print(int((hr.shape[0]-h*32)/2), int(hr.shape[0]-(hr.shape[0]-h*32)/2))
        #     # assert(1==2)
        #     hr = hr[int((hr.shape[0]-h*32)/2):int(hr.shape[0]-(hr.shape[0]-h*32)/2), int((hr.shape[1]-w*32)/2):int(hr.shape[1]-(hr.shape[1]-w*32)/2), :]
        #     lr = lr[int((hr.shape[0]-h*32)/2):int(hr.shape[0]-(hr.shape[0]-h*32)/2), int((hr.shape[1]-w*32)/2):int(hr.shape[1]-(hr.shape[1]-w*32)/2), :]

        # if self.center_crop_hr_size:
        #     hr, lr = center_crop(hr, self.center_crop_hr_size), center_crop(lr, self.center_crop_hr_size)

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)


        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        # print(f_name, hr.shape, lr.shape)

        [lr, hr] = transform_augment(
                [lr, hr], split=self.split, min_max=(-1, 1))
        # print(f_name)
        return {'LQ': lr, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name}


class Rain200L_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("patch_size", None)
        if train:
            self.split = 'train'
            self.root = os.path.join(self.root, 'train')
        else:
            self.split = 'val'
            self.root = os.path.join(self.root, 'test')
        self.pairs = self.load_pairs(self.root)
        self.to_tensor = ToTensor()
        

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):

        low_list = os.listdir(os.path.join(folder_path, 'low'))
        # low_list = natsort.natsorted(os.listdir(os.path.join(folder_path, 'low')),alg = natsort.ns.PATH)

        # low_list.sort(key=lambda x:int(x.split('.')[0]))
        low_list = filter(lambda x: 'jpg' or 'png' in x, low_list)
        
        
        pairs = []
        for idx, f_name in enumerate(low_list):
            # print(idx, f_name)
            if self.split == 'val':
                # print(f_name, f_name[0:len(f_name)-9]+'_clean.png')
                pairs.append(
                    [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB),
                    #  cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name.split('_')[0]+'.jpg')), cv2.COLOR_BGR2RGB), 
                     cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name)), cv2.COLOR_BGR2RGB), 
                    #  cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name.split('x')[0]+'.png')), cv2.COLOR_BGR2RGB),
                     cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low1', 'norain-'+f_name.split('.')[0]+'x2.png')), cv2.COLOR_BGR2RGB),
                    f_name.split('.')[0]])
            else:
                # print(f_name)
                pairs.append(
                    [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB), 
                    #  cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name.split('_')[0]+'.jpg')), cv2.COLOR_BGR2RGB), 
                    #  cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name)), cv2.COLOR_BGR2RGB), 
                     cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name.split('x')[0]+'.png')), cv2.COLOR_BGR2RGB),
                     cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name.split('x')[0]+'.png')), cv2.COLOR_BGR2RGB),

                    f_name.split('.')[0]])
        return pairs

    def __getitem__(self, item):
            
        lr, hr, lr2, f_name = self.pairs[item]
        # print('item', item, index, f_name)
        # hr = hr[0:480, 0:720, :]
        # lr = lr[0:480, 0:720, :]
        # print(f_name, hr.shape, lr.shape, self.crop_size)


        if self.use_crop and self.split != 'val':
            hr, lr = random_crop(hr, lr, self.crop_size)
        elif self.split == 'val':
            lr = cv2.copyMakeBorder(lr, 0,7,0,7, cv2.BORDER_REFLECT)
            lr2 = cv2.copyMakeBorder(lr2, 0,7,0,7, cv2.BORDER_REFLECT)

            # if lr.shape[0] > lr.shape[1]:
            #     hr = hr[0:480, 0:320, :]
            #     lr = lr[0:480, 0:320, :]
            # else:
            #     hr = hr[0:320, 0:480, :]
            #     lr = lr[0:320, 0:480, :]                
            # print(hr.shape)
            # lr = cv2.resize(lr, (480, 480))
            # hr = cv2.resize(hr, (480, 480))
            # lr = cv2.resize(lr, (lr.shape[1] // 16 * 16, lr.shape[0] // 16 * 16))
            # hr = cv2.resize(hr, (hr.shape[1] // 16 * 16, hr.shape[0] // 16 * 16))
            # print(hr.shape)

        # elif self.use_crop and self.split == 'val':
        #     h = int(hr.shape[0]/32)
        #     w = int(hr.shape[1]/32)
        #     # print(hr.shape)
        #     # print(h,w)
        #     # print(int((hr.shape[0]-h*32)/2), int(hr.shape[0]-(hr.shape[0]-h*32)/2))
        #     # assert(1==2)
        #     hr = hr[int((hr.shape[0]-h*32)/2):int(hr.shape[0]-(hr.shape[0]-h*32)/2), int((hr.shape[1]-w*32)/2):int(hr.shape[1]-(hr.shape[1]-w*32)/2), :]
        #     lr = lr[int((hr.shape[0]-h*32)/2):int(hr.shape[0]-(hr.shape[0]-h*32)/2), int((hr.shape[1]-w*32)/2):int(hr.shape[1]-(hr.shape[1]-w*32)/2), :]

        # if self.center_crop_hr_size:
        #     hr, lr = center_crop(hr, self.center_crop_hr_size), center_crop(lr, self.center_crop_hr_size)

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)


        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)
        lr2 = self.to_tensor(lr2)
        

        # print(f_name, hr.shape, lr.shape)

        [lr, hr, lr2] = transform_augment(
                [lr, hr, lr2], split=self.split, min_max=(-1, 1))
        
        # Compute rain streak image (Rain = LQ - GT)
        # rain = torch.clamp(lr - hr, min=-1, max=1)  # Ensure values stay in range [-1, 1]
        # rain_weight = torch.where(rain > 0, 10.0, 1.0)  # 雨条区域权重大
        
        # self.save_rain_image(rain, f_name)
        # print(f_name)
        return {'LQ': lr, 'GT': hr, 'LQ2': lr2,'LQ_path': f_name, 'GT_path': f_name}

    def save_rain_image(self, rain, file_name):
        # Define the save directory
        save_dir = os.path.join('/data1/hou_21/project/LLIE/GSAD-v3/data', 'rain_images')
        os.makedirs(save_dir, exist_ok=True)

        # Denormalize rain image to [0, 255]
        rain = (rain + 1.0) * 127.5  # Scale back to [0, 255]
        rain = rain.permute(1, 2, 0).cpu().numpy().astype('uint8')  # Convert to NumPy format

        # Save as an image
        rain_image = Image.fromarray(rain)
        rain_image.save(os.path.join(save_dir, f'{file_name}_rain.png'))

class Rain200H_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("patch_size", None)
        if train:
            self.split = 'train'
            self.root = os.path.join(self.root, 'train')
        else:
            self.split = 'val'
            self.root = os.path.join(self.root, 'test')
        self.pairs = self.load_pairs(self.root)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):

        low_list = os.listdir(os.path.join(folder_path, 'low'))
        low_list = filter(lambda x: 'jpg' or 'png' in x, low_list)

        pairs = []
        for idx, f_name in enumerate(low_list):
            
            if self.split == 'val':
                # print(f_name, f_name[0:len(f_name)-9]+'_clean.png')
                pairs.append(
                    [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB),  
                     cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name)), cv2.COLOR_BGR2RGB), 
                    #  cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name.split('x')[0]+'.png')), cv2.COLOR_BGR2RGB),
                     cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low1', f_name)), cv2.COLOR_BGR2RGB),

                    f_name.split('.')[0]])
            else:
                # print(f_name)
                pairs.append(
                    [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB),  
                    #  cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name)), cv2.COLOR_BGR2RGB), 
                     cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name.split('x')[0]+'.png')), cv2.COLOR_BGR2RGB),
                     cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name.split('x')[0]+'.png')), cv2.COLOR_BGR2RGB),
                    f_name.split('.')[0]])
        return pairs

    def __getitem__(self, item):
        lr, hr, lr2, f_name = self.pairs[item]
        
        # hr = hr[0:480, 0:720, :]
        # lr = lr[0:480, 0:720, :]
        # print(f_name, hr.shape, lr.shape, self.crop_size)
        

        if self.use_crop and self.split != 'val':
            hr, lr = random_crop(hr, lr, self.crop_size)
        elif self.split == 'val':
            lr = cv2.copyMakeBorder(lr, 0,7,0,7, cv2.BORDER_REFLECT)
            lr2 = cv2.copyMakeBorder(lr2, 0,7,0,7, cv2.BORDER_REFLECT)
            # print(lr.shape)
            # print(hr.shape)
            # lr = cv2.resize(lr, (lr.shape[1] // 16 * 16, lr.shape[0] // 16 * 16))
            # hr = cv2.resize(hr, (hr.shape[1] // 16 * 16, hr.shape[0] // 16 * 16))
            # lr = cv2.resize(lr, (lr.shape[0] // 16 * 16, lr.shape[1] // 16 * 16))
            # hr = cv2.resize(hr, (hr.shape[0] // 16 * 16, hr.shape[1] // 16 * 16))
            # print(hr.shape)

        # elif self.use_crop and self.split == 'val':
        #     h = int(hr.shape[0]/32)
        #     w = int(hr.shape[1]/32)
        #     # print(hr.shape)
        #     # print(h,w)
        #     # print(int((hr.shape[0]-h*32)/2), int(hr.shape[0]-(hr.shape[0]-h*32)/2))
        #     # assert(1==2)
        #     hr = hr[int((hr.shape[0]-h*32)/2):int(hr.shape[0]-(hr.shape[0]-h*32)/2), int((hr.shape[1]-w*32)/2):int(hr.shape[1]-(hr.shape[1]-w*32)/2), :]
        #     lr = lr[int((hr.shape[0]-h*32)/2):int(hr.shape[0]-(hr.shape[0]-h*32)/2), int((hr.shape[1]-w*32)/2):int(hr.shape[1]-(hr.shape[1]-w*32)/2), :]

        # if self.center_crop_hr_size:
        #     hr, lr = center_crop(hr, self.center_crop_hr_size), center_crop(lr, self.center_crop_hr_size)

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)


        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)
        lr2 = self.to_tensor(lr2)

        # if self.split == 'val':
        #     img_multiple_of = 8
        #     print(lr.shape)
        #     height,width = lr.shape[1], lr.shape[2]
        #     H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        #     padh = H-height if height%img_multiple_of!=0 else 0
        #     padw = W-width if width%img_multiple_of!=0 else 0
        #     print(padh, padw)
        #     lr = F.pad(lr, (0,padw,0,padh), 'reflect')
        #     print(lr.shape)
        # print(f_name, hr.shape, lr.shape)

        [lr, hr, lr2] = transform_augment(
                [lr, hr, lr2], split=self.split, min_max=(-1, 1))

        return {'LQ': lr, 'LQ2': lr2, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name}


class RainHeavy_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("GT_size", None)
        if train:
            self.split = 'train'
            self.root = os.path.join(self.root, 'train')
        else:
            self.split = 'val'
            self.root = os.path.join(self.root, 'train')
        self.pairs = self.load_pairs(self.root)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):

        low_list = os.listdir(os.path.join(folder_path, 'low'))
        low_list = filter(lambda x: 'png' in x, low_list)

        
        testlist = []
        with open("./data/heavy_rain_test1.txt", "r") as f:
            data = f.readlines()
            for str in data:
                if len(str)>5:
                    str = str[:-1]
                else:
                    str = str
                testlist.append(str)
            # print(testlist, len(testlist))
        pairs = []
        if self.split == 'val':
            for f_name in testlist: 
                pairs.append(
                    [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB),  
                        cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name[0:7]+'.png')), cv2.COLOR_BGR2RGB),
                    f_name.split('.')[0]])
            # print('testlist', len(pairs))
        else:
            for idx, f_name in enumerate(low_list):
                if f_name in testlist:
                    continue
                print(idx)
                pairs.append(
                    [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB),  
                    cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name[0:7]+'.png')), cv2.COLOR_BGR2RGB),
                    f_name.split('.')[0]])
            # print('trainlist', len(pairs))
        return pairs

    def __getitem__(self, item):
        lr, hr, f_name = self.pairs[item]
        
        hr = hr[0:480, 0:720, :]
        lr = lr[0:480, 0:720, :]
        # print(f_name, hr.shape, lr.shape)

        if self.use_crop and self.split != 'val':
            hr, lr = random_crop(hr, lr, self.crop_size)
        # elif self.use_crop and self.split == 'val':
        #     hr = hr[8:392, 12:588, :]
        #     lr = lr[8:392, 12:588, :]

        # elif self.use_crop and self.split == 'val':
        #     h = int(hr.shape[0]/32)
        #     w = int(hr.shape[1]/32)
        #     # print(hr.shape)
        #     # print(h,w)
        #     # print(int((hr.shape[0]-h*32)/2), int(hr.shape[0]-(hr.shape[0]-h*32)/2))
        #     # assert(1==2)
        #     hr = hr[int((hr.shape[0]-h*32)/2):int(hr.shape[0]-(hr.shape[0]-h*32)/2), int((hr.shape[1]-w*32)/2):int(hr.shape[1]-(hr.shape[1]-w*32)/2), :]
        #     lr = lr[int((hr.shape[0]-h*32)/2):int(hr.shape[0]-(hr.shape[0]-h*32)/2), int((hr.shape[1]-w*32)/2):int(hr.shape[1]-(hr.shape[1]-w*32)/2), :]

        # if self.center_crop_hr_size:
        #     hr, lr = center_crop(hr, self.center_crop_hr_size), center_crop(lr, self.center_crop_hr_size)

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)


        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        # print(f_name, hr.shape, lr.shape)

        [lr, hr] = transform_augment(
                [lr, hr], split=self.split, min_max=(-1, 1))

        return {'LQ': lr, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name}


class Raindrop_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("GT_size", None)
        if train:
            self.split = 'train'
            self.root = os.path.join(self.root, 'train')
        else:
            self.split = 'val'
            self.root = os.path.join(self.root, 'test_a58')
        self.pairs = self.load_pairs(self.root)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):

        low_list = os.listdir(os.path.join(folder_path, 'low'))
        low_list = filter(lambda x: 'png' in x, low_list)

        pairs = []
        for idx, f_name in enumerate(low_list):
            
            if self.split == 'val':
                pairs.append(
                    [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB),  
                     cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name[0:len(f_name)-9]+'_clean.png')), cv2.COLOR_BGR2RGB),
                    f_name.split('.')[0]])
            else:
                # print(f_name, f_name[0:len(f_name)-9]+'_clean.png')
                pairs.append(
                    [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB),  
                     cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name[0:len(f_name)-9]+'_clean.png')), cv2.COLOR_BGR2RGB),
                    f_name.split('.')[0]])
        return pairs

    def __getitem__(self, item):
        lr, hr, f_name = self.pairs[item]
        
        hr = hr[0:480, 0:720, :]
        lr = lr[0:480, 0:720, :]

        if self.use_crop and self.split != 'val':
            hr, lr = random_crop(hr, lr, self.crop_size)
        # elif self.use_crop and self.split == 'val':
        #     hr = hr[8:392, 12:588, :]
        #     lr = lr[8:392, 12:588, :]


        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)

        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        [lr, hr] = transform_augment(
                [lr, hr], split=self.split, min_max=(-1, 1))

        return {'LQ': lr, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name}

class UnderwaterB_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("GT_size", None)
        if train:
            self.split = 'train'
            self.root = os.path.join(self.root, 'train')
        else:
            self.split = 'val'
            self.root = os.path.join(self.root, 'train')
        self.pairs = self.load_pairs(self.root)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):

        low_list = os.listdir(os.path.join(folder_path, 'raw-890'))
        low_list = filter(lambda x: 'png' in x, low_list)

        
        testlist = []
        with open("./data/uie_test_list.txt", "r") as f:
            data = f.readlines()
            for str in data:
                if len(str)>5:
                    str = str[:-1]
                else:
                    str = str
                testlist.append(str)
            # print(testlist, len(testlist))
        pairs = []
        if self.split == 'val':
            for f_name in testlist[0:90]: 
                
                pairs.append(
                    [os.path.join(folder_path, 'raw-890', f_name),  
                     os.path.join(folder_path, 'reference-890', f_name),
                     f_name.split('.')[0]])
        else:
            for idx, f_name in enumerate(low_list):
                if f_name in testlist:
                    continue
                print(idx)
                pairs.append(
                    [os.path.join(folder_path, 'raw-890', f_name),  
                     os.path.join(folder_path, 'reference-890', f_name),
                     f_name.split('.')[0]])
        return pairs

    def __getitem__(self, item):
        raw_img_path, gt_img_path, f_name = self.pairs[item]
        raw_img = Image.open(raw_img_path)
        gt_img = Image.open(gt_img_path)
        img_w = gt_img.size[0]
        img_h = gt_img.size[1]
        # print(f_name, hr.shape, lr.shape)

        if self.split == 'train':

            # ### data process 1
            # i, j, h, w = transforms.RandomResizedCrop(self.crop_size).get_params(raw_img, (0.08, 1.0),(3. / 4., 4. / 3.))   
            # raw_cropped = F.resized_crop(raw_img, i, j, h, w, (self.crop_size, self.crop_size), InterpolationMode.BICUBIC)
            # gt_cropped = F.resized_crop(gt_img, i, j, h, w, (self.crop_size, self.crop_size), InterpolationMode.BICUBIC)

            # raw_cropped = transforms.ToTensor()(raw_cropped)
            # gt_cropped = transforms.ToTensor()(gt_cropped)

            # if np.random.rand(1) < 0.5:  # flip horizonly
            #     raw_cropped = torch.flip(raw_cropped, [2])
            #     gt_cropped = torch.flip(gt_cropped, [2])
            # if np.random.rand(1) < 0.5:  # flip vertically
            #     raw_cropped = torch.flip(raw_cropped, [1])
            #     gt_cropped = torch.flip(gt_cropped, [1])

            ### data process 2
            raw_cropped = Image.open(raw_img_path).convert("RGB")
            gt_cropped = Image.open(gt_img_path).convert("RGB")

            raw_cropped = raw_cropped.resize((320, 320), Image.ANTIALIAS)
            gt_cropped = gt_cropped.resize((320, 320), Image.ANTIALIAS)
            w, h = raw_cropped.size
            x, y = random.randrange(w - self.crop_size + 1), random.randrange(h - self.crop_size + 1)
            raw_cropped = raw_cropped.crop((x, y, x + self.crop_size, y + self.crop_size))
            gt_cropped = gt_cropped.crop((x, y, x + self.crop_size, y + self.crop_size))

            raw_cropped = transforms.ToTensor()(raw_cropped)
            gt_cropped = transforms.ToTensor()(gt_cropped)

            if np.random.rand(1) < 0.5:  # flip horizonly
                raw_cropped = torch.flip(raw_cropped, [2])
                gt_cropped = torch.flip(gt_cropped, [2])
            if np.random.rand(1) < 0.5:  # flip vertically
                raw_cropped = torch.flip(raw_cropped, [1])
                gt_cropped = torch.flip(gt_cropped, [1])

            raw_img = raw_cropped
            gt_img = gt_cropped
            
        elif self.split == "val":
            # raw_img = transforms.Resize((img_h // 16 * 16, img_w // 16 * 16))(raw_img)
            raw_img = transforms.Resize((320, 320))(raw_img)
            
            raw_img = transforms.ToTensor()(raw_img)
            # gt_img = transforms.Resize((img_h // 32 * 32, img_w // 32 * 32))(gt_img)
            gt_img = transforms.ToTensor()(gt_img)
            
        # print(f_name, gt_img.shape, raw_img.shape)

        [raw_img, gt_img] = transform_augment(
                [raw_img, gt_img], split=self.split, min_max=(-1, 1))

        return {'LQ': raw_img, 'GT': gt_img, 'LQ_path': f_name, 'GT_path': f_name}


class LOLv1_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.crop_size = opt.get("patch_size", None)
        if train:
            self.split = 'train'
            self.root = os.path.join(self.root, 'our485')
        else:
            self.split = 'val'
            self.root = os.path.join(self.root, 'eval15')
        self.pairs = self.load_pairs(self.root)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):

        low_list = os.listdir(os.path.join(folder_path, 'low'))
        low_list = filter(lambda x: 'png' in x, low_list)

        pairs = []
        for idx, f_name in enumerate(low_list):
            
            if self.split == 'val':
                # print(os.path.join(folder_path, 'mid', f_name))
                # print(os.path.join(folder_path, 'high', f_name))
                f_name1 = f_name
                # f_name2 = f_name.replace('lq', 'gt')
                # f_name3 = f_name.replace('lq', 'normal_noadjust')
                pairs.append(
                    [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB),  
                     cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name)), cv2.COLOR_BGR2RGB),
                    f_name.split('.')[0]])
            else:
                pairs.append(
                    [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB),  
                     cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name)), cv2.COLOR_BGR2RGB),
                    f_name.split('.')[0]])
        return pairs

    def __getitem__(self, item):
        lr, hr, f_name = self.pairs[item]


        if self.use_crop and self.split != 'val':
            hr, lr = random_crop(hr, lr, self.crop_size)
        elif self.split == 'val':
            lr = cv2.copyMakeBorder(lr, 8,8,4,4,cv2.BORDER_REFLECT)
            # hr = cv2.copyMakeBorder(hr, 8,8,4,4,cv2.BORDER_REFLECT)

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)

        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        [lr, hr] = transform_augment(
                [lr, hr], split=self.split, min_max=(-1, 1))

        return {'LQ': lr, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name}

class LOLv2_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.crop_size = opt.get("patch_size", None)
        self.sub_data = opt.get("sub_data", None)
        self.pairs = []
        self.train = train
        if train:
            self.split = 'train'
            root = os.path.join(self.root, self.sub_data, 'Train')
        else:
            self.split = 'val'
            root = os.path.join(self.root, self.sub_data, 'Test-all')
        self.pairs.extend(self.load_pairs(root))
        self.to_tensor = ToTensor()
        self.gamma_aug = opt['gamma_aug'] if 'gamma_aug' in opt.keys() else False

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):

        low_list = os.listdir(os.path.join(folder_path, 'Low' if self.train else 'Low'))
        low_list = sorted(list(filter(lambda x: 'png' in x, low_list)))
        
        high_list = os.listdir(os.path.join(folder_path, 'Normal' if self.train else 'Normal'))
        high_list = sorted(list(filter(lambda x: 'png' in x, high_list)))
        pairs = []

        for idx in range(len(low_list)):
            f_name_low = low_list[idx]
            f_name_high = high_list[idx]
            pairs.append(
                [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'Low' if self.train else 'Low', f_name_low)),
                                cv2.COLOR_BGR2RGB),  
                    cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'Normal' if self.train else 'Normal', f_name_high)),
                                cv2.COLOR_BGR2RGB), 
                    f_name_high.split('.')[0]])
        return pairs

    def __getitem__(self, item):
        
        lr, hr, f_name = self.pairs[item]

        if self.use_crop and self.split != 'val':
            hr, lr = random_crop(hr, lr, self.crop_size)
        elif self.sub_data == 'Real_captured' and self.split == 'val': # for Real_captured
            lr = cv2.copyMakeBorder(lr, 8,8,4,4,cv2.BORDER_REFLECT)

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)


        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        
        [lr, hr] = transform_augment(
                [lr, hr], split=self.split, min_max=(-1, 1))

        return {'LQ': lr, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name}


def random_flip(img, seg):
    random_choice = np.random.choice([True, False])
    img = img if random_choice else np.flip(img, 1).copy()
    seg = seg if random_choice else np.flip(seg, 1).copy()

    return img, seg


def gamma_aug(img, gamma=0):
    max_val = img.max()
    img_after_norm = img / max_val
    img_after_norm = np.power(img_after_norm, gamma)
    return img_after_norm * max_val


def random_rotation(img, seg):
    random_choice = np.random.choice([0, 1, 3])
    img = np.rot90(img, random_choice, axes=(0, 1)).copy()
    seg = np.rot90(seg, random_choice, axes=(0, 1)).copy()
    
    return img, seg


def random_crop(hr, lr, size_hr):
    size_lr = size_hr

    size_lr_x = lr.shape[0]
    size_lr_y = lr.shape[1]

    start_x_lr = np.random.randint(low=0, high=(size_lr_x - size_lr) + 1) if size_lr_x > size_lr else 0
    start_y_lr = np.random.randint(low=0, high=(size_lr_y - size_lr) + 1) if size_lr_y > size_lr else 0

    # LR Patch
    lr_patch = lr[start_x_lr:start_x_lr + size_lr, start_y_lr:start_y_lr + size_lr, :]

    # HR Patch
    start_x_hr = start_x_lr
    start_y_hr = start_y_lr
    hr_patch = hr[start_x_hr:start_x_hr + size_hr, start_y_hr:start_y_hr + size_hr, :]

    # HisEq Patch
    his_eq_patch = None
    return hr_patch, lr_patch, 


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
def transform_augment(imgs, split='val', min_max=(0, 1)):    
    # imgs = [totensor(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        # imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img


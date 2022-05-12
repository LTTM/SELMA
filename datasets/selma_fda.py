import json
from os import path
import numpy as np
from plyfile import PlyData
import cv2 as cv
from datasets.cityscapes import CityDataset
from datasets.carlaLTTM import LTTMDataset
from torch.utils.data import Dataset
from math import ceil
from torch.fft import fft2, ifft2
import torch

class SELMA_FDA(Dataset):
    def __init__(self, beta=.005, resize_to=(1024,512)):
        self.selma = LTTMDataset(root_path = "D:/Datasets/SELMA/data",
                                 splits_path = "D:/Datasets/SELMA/splits",
                                 split = "train_rand",
                                 sensors=['rgb', 'semantic'],
                                 augment_data=False,
                                 resize_to=resize_to)
        self.city = CityDataset(root_path = "F:/Dataset/Cityscapes_extra",
                                 splits_path = "F:/Dataset/Cityscapes_extra",
                                 sensors=['rgb', 'semantic'],
                                 augment_data=False,
                                 resize_to=resize_to)
        self.ratio = ceil(len(self.selma)/len(self.city))
        self.beta = beta
        self.mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
 
    def __len__(self):
        return min(len(self.selma), len(self.city))

    def __getitem__(self, item):
        it_range = np.random.randint(self.ratio)
        sitem = (it_range*len(self.city)+item)%len(self.selma)
        ss = self.selma[sitem][0]
        sc = self.city[item][0]
        
        rgb_s, rgb_c_og, gt = ss['rgb']['D'], sc['rgb'], ss['semantic']['D']
        rgb_s = rgb_s*self.mean+self.mean
        rgb_c = rgb_c_og*self.mean+self.mean
        H, W = rgb_s.shape[1:]
        h, w = ceil(self.beta*H), ceil(self.beta*W)
        
        ffss = fft2(rgb_s)
        ffsc = fft2(rgb_c).abs()

        ffss_m, ffss_a = ffss.abs(), ffss.angle()

        ffss_m[...,:h,:w] = ffsc[...,:h,:w]
        ffss_m[...,:h,-w:] = ffsc[...,:h,-w:]
        ffss_m[...,-h:,:w] = ffsc[...,-h:,:w]
        ffss_m[...,-h:,-w:] = ffsc[...,-h:,-w:]
        
        if np.random.rand() < .5:
            scale = np.random.randint(20)+10
            ffss_m[...,scale*h:-scale*h,scale*w:-scale*w] = 0

        ffss = torch.polar(ffss_m, ffss_a)
        rgb = torch.clamp(ifft2(ffss).real, 0, 1)
        rgb = (rgb-self.mean)/self.mean
        
        if np.random.rand() < .5:
            rgb = torch.flip(rgb, dims=(-1,))
            gt = torch.flip(gt, dims=(-1,))
        
        return rgb, gt, rgb_c_og
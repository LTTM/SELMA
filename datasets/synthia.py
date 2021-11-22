from torch.utils.data import Dataset
import cv2 as cv
from PIL import Image
import imageio
import numpy as np
import torch
from plyfile import PlyData
from os import path, listdir
from utils.cmaps import City36cmap, City19cmap, City16cmap, City13cmap

class SynthiaDataset(Dataset):
    def __init__(self,
                 root_path="F:/Dataset/SYNTHIA/full/RAND_CITYSCAPES",
                 split='train_tot',
                 sensors=['rgb', 'semantic']): # whether to use city19 or city36 class set
         
        assert type(sensors) is list, "sensors must be a list"
        for s in sensors:
            assert s in ['rgb','semantic'], "Unrecognized Sensor: "+s+", must be one of ['rgb', 'semantic']"
         
        self.root_path = root_path
        self.sensors = sensors
        with open(path.join(root_path,split+'.txt')) as f:
            self.items = [l.rstrip('\n').split(' ') for l in f]
            self.items = [(t[0].lstrip('/'), t[1].lstrip('/')) for t in self.items]
        
        
        self.cmap = City16cmap
        self.raw2train = {1:9, 2:2, 3:0, 4:1, 5:4, 6:8, 7:5, 8:12, 9:7, 10:10,
                          11:15, 12:14, 15:6, 16:-1, 17:11, 18:-1, 19:13, 20:-1, 21:3}
        self.ignore_index = -1
    
    def __getitem__(self, item):
        rgb, gt = self.items[item]
        out_dict = {}
        if 'rgb' in self.sensors:
            out_dict['rgb'] = self.to_pytorch(self.load_rgb(path.join(self.root_path, rgb)))
        if 'semantic' in self.sensors:
            out_dict['semantic'] = self.load_semantic(path.join(self.root_path, gt))

        return out_dict, item
                    
    def __len__(self):
        return len(self.items)
        
    @staticmethod
    def load_rgb(path):
        return cv.imread(path).copy()
        
    @staticmethod
    def to_pytorch(bgr):
        bgr = np.transpose(bgr.astype(np.float32)-[104.00698793, 116.66876762, 122.6789143], (2, 0, 1))
        return torch.from_numpy(bgr)
        
    def load_semantic(self, path):
        gt = np.array(imageio.imread(path, format='PNG-FI')[:, :, 0]).astype(np.uint8)
        if self.raw2train is not None:
            out = self.ignore_index*np.ones_like(gt, dtype=int)
            for k,v in self.raw2train.items():
                out[gt==k] = v
        else:
            out = gt.copy()
        return out
        
    def color_labels(self, label):
        assert len(label.shape) < 3, 'Input label must either be a grayscale image or a 1-dimensional array'
        return self.cmap[label]
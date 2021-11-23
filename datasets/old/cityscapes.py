from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
import torch
from plyfile import PlyData
from os import path, listdir
from utils.cmaps import City36cmap, City19cmap

class CityDataset(Dataset):
    def __init__(self,
                 root_path="F:/Dataset/Cityscapes_extra/",
                 split='alltrain',
                 resize_to=None,
                 sensors=['rgb', 'semantic'],
                 extended_classes=False): # whether to use city19 or city36 class set
         
        assert type(sensors) is list, "sensors must be a list"
        for s in sensors:
            assert s in ['rgb','semantic'], "Unrecognized Sensor: "+s+", must be one of ['rgb', 'semantic']"
        
        self.resize_to = resize_to
        self.root_path = root_path
        self.sensors = sensors
        with open(path.join(root_path,split+'.txt')) as f:
            self.items = [l.rstrip('\n').split(' ') for l in f]
                    
        if extended_classes:
            self.cmap = City36cmap
            self.raw2train = None
            self.ignore_index = 0
        else:
            self.cmap = City19cmap
            self.raw2train = {0:-1, 1:-1, 2:-1, 3:-1, 4:-1, 5:-1, 6:-1, 7:0, 8:1, 9:-1, 10:-1, 
                              11:2, 12:3, 13:4, 14:-1, 15:-1, 16:-1, 17:5, 18:-1, 19:6, 20:7,
                              21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:-1, 30:-1,
                              31:16, 32:17, 33:18, 34:-1, 35:-1}
            self.ignore_index = -1
    
    def __getitem__(self, item):
        rgb, gt = self.items[item]
        out_dict = {}
        if 'rgb' in self.sensors:
            out_dict['rgb'] = self.to_pytorch(self.resize(self.load_rgb(path.join(self.root_path, rgb))))
        if 'semantic' in self.sensors:
            out_dict['semantic'] = self.resize(self.load_semantic(path.join(self.root_path, gt)), True)

        return out_dict, item
                    
    def __len__(self):
        return len(self.items)
        
    @staticmethod
    def load_rgb(path):
        return cv.imread(path).copy()
        
    @staticmethod
    def to_rgb(tensor):
        t = np.array(tensor.transpose(0,1).transpose(1,2))+[104.00698793, 116.66876762, 122.6789143] # bgr
        t = np.round(t[...,::-1]).astype(np.uint8) # rgb
        return t
        
    @staticmethod
    def to_pytorch(bgr):
        bgr = np.transpose(bgr.astype(np.float32)-[104.00698793, 116.66876762, 122.6789143], (2, 0, 1))
        return torch.from_numpy(bgr)
        
    def load_semantic(self, path):
        gt = cv.imread(path, cv.IMREAD_UNCHANGED)
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
        
    def resize(self, im, gt=False):
        if self.resize_to is not None:
            if not gt:
                im = cv.resize(im, self.resize_to, interpolation=cv.INTER_AREA) # usually images are downsized, best results obtained with inter_area
                return im
            else:
                im = cv.resize(im, self.resize_to, interpolation=cv.INTER_NEAREST_EXACT) # labels must be mapped as-is
                return im.astype(np.long)
        else:
            return im
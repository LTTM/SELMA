from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
import torch
from plyfile import PlyData
from os import path, listdir
from utils.cmaps import City36cmap, City19cmap

class IDDDataset(Dataset):
    def __init__(self,
                 root_path="F:/Dataset/IDD",
                 split='trainval',
                 sensors=['rgb', 'semantic'],
                 extended_classes=False): # whether to use city19 or city36 class set
         
        assert type(sensors) is list, "sensors must be a list"
        for s in sensors:
            assert s in ['rgb','semantic'], "Unrecognized Sensor: "+s+", must be one of ['rgb', 'semantic']"
         
        self.root_path = root_path
        self.sensors = sensors
        with open(path.join(root_path,split+'.txt')) as f:
            self.items = [l.rstrip('\n').split(' ') for l in f]
                    
        if extended_classes:
            self.cmap = City36cmap
            self.raw2train = {0:7, 1:9, 2:7, 3:8, 4:6, 5:10, 6:24, 7:5, 8:25, 9:32, 10:33, 11:5, 12:26,
                              13:27, 14:28, 15:29, 16:30, 17:31, 18:5, 19:4, 20:12, 21:13, 22:14, 23:4,
                              24:20, 25:19, 26:17, 27:18, 28:4, 29:11, 30:15, 31:16, 32:21, 33:23, 34:6,
                              255:0, 251:1, 252:34, 253:2, 254:3}
            self.ignore_index = 0
        else:
            self.cmap = City19cmap
            self.raw2train = {0:0, 1:-1, 2:-1, 3:1, 4:-1, 5:9, 6:11, 7:-1, 8:12, 9:17, 10:18, 11:-1,
                              12:13, 13:14, 14:15, 15:-1, 16:-1, 17:16, 18:-1, 19:-1, 20:3, 21:4,
                              22:-1, 23:-1, 24:7, 25:6, 26:5, 27:-1, 28:-1, 29:2, 30:-1, 31:-1,
                              32:8, 33:10, 34:-1, 255:-1, 251:-1, 252:-1, 253:-1, 254:-1}
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
        
"""
IDD Labels -> City36

road                      0->    7      road
parking                   1->    9      parking
drivable fallback         2->  0/7      unlabeled/road
sidewalk                  3->    8      sidewalk
non-drivable fallback     4->  0/6      unlabeled/ground
rail track                5->   10      rail track
person                    6->   24      person
animal                    7->  0/5      unlabeled/dynamic
rider                     8->   25      rider
motorcycle                9->   32      motorcycle
bicycle                  10->   33      bicycle
autorickshaw             11->  0/5      unlabeled/dynamic
car                      12->   26      car
truck                    13->   27      truck
bus                      14->   28      bus
caravan                  15->   29      caravan
trailer                  16->   30      trailer    
train                    17->   31      train
vehicle fallback         18->  0/5      unlabeled/dynamic
curb                     19->  0/4      unlabeled/static
wall                     20->   12      wall
fence                    21->   13      fence
guard rail               22->   14      guard rail
billboard                23->    4      static
traffic sign             24->   20      traffic sign
traffic light            25->   19      traffic light
pole                     26->   17      pole
polegroup                27->   18      polegroup
obs-str-bar-fallback     28->    4      static
building                 29->   11      building
bridge                   30->   15      bridge
tunnel                   31->   16      tunnel
vegetation               32->   21      vegetation
sky                      33->   23      sky
fallback background      34->  0/6      unlabeled/ground
unlabeled               255->    0      unlabeled
ego vehicle             251->    1      ego vehicle
license plate           252->   34      license plate
rectification border    253->    2      rectification border
out of roi              254->    3      out of roi
"""
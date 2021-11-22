from torch.utils.data import Dataset
import cv2 as cv
from PIL import Image
import numpy as np
import torch
from plyfile import PlyData
from os import path, listdir
from utils.cmaps import City36cmap, City19cmap

class MapillaryDataset(Dataset):
    def __init__(self,
                 root_path="F:/Dataset/Mapillary",
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
            self.raw2train = {0:5, 1:5, 2:4, 3:13, 4:14, 5:4, 6:12, 7:7, 8:7, 9:4, 10:9, 11:6,
                              12:10, 13:7, 14:7, 15:8, 16:15, 17:11, 18:16, 19:24, 20:25, 21:25, 22:25,
                              23:7, 24:7, 25:22, 26:22, 27:23, 28:22, 29:22, 30:21, 31:35, 32:4, 33:4,
                              34:4, 35:4, 36:4, 37:4, 38:4, 39:4, 40:4, 41:4, 42:4, 43:4, 44:4, 45:17,
                              46:4, 47:17, 48:19, 49:20, 50:20, 51:4, 52:33, 53:5, 54:28, 55:26,
                              56:29, 57:32, 58:31, 59:5, 60:30, 61:27, 62:5, 63:1, 64:1, 65:0}
            self.ignore_index = 0
        else:
            self.cmap = City19cmap
            self.raw2train = {0:-1, 1:-1, 2:-1, 3:4, 4:-1, 5:-1, 6:3, 7:0, 8:0, 9:-1, 10:-1, 11:-1, 12:-1,
                              13:0, 14:0, 15:1, 16:-1, 17:2, 18:-1, 19:11, 20:12, 21:12, 22:12, 23:0, 24:0,
                              25:9, 26:9, 27:10, 28:9, 29:9, 30:8, 31:-1, 32:-1, 33:-1, 34:-1, 35:-1, 36:-1,
                              37:-1, 38:-1, 39:-1, 40:-1, 41:-1, 42:-1, 43:-1, 44:-1, 45:5, 46:-1, 47:5,
                              48:6, 49:7, 50:7, 51:-1, 52:18, 53:-1, 54:15, 55:13, 56:-1, 57:17, 58:16, 59:-1,
                              60:-1, 61:14, 62:-1, 63:-1, 64:-1, 65:-1}
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
        gt = np.array(Image.open(path))
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
0: 'Bird', -> dynamic 5
1: 'Ground Animal', -> dynamic 5
2: 'Curb', -> static 4
3: 'Fence', -> fence 13
4: 'Guard Rail', -> guard rail 14
5: 'Barrier', -> static 4
6: 'Wall', -> wall 12
7: 'Bike Lane', -> road 7
8: 'Crosswalk - Plain', -> road 7
9: 'Curb Cut', -> static 4
10: 'Parking', -> parking 9
11: 'Pedestrian Area', -> ground 6
12: 'Rail Track', -> rail track 10
13: 'Road', -> road 7
14: 'Service Lane', -> road 7
15: 'Sidewalk', -> sidewalk 8
16: 'Bridge', -> bridge 15
17: 'Building', -> building 11
18: 'Tunnel', -> tunnel 16
19: 'Person', -> person 24
20: 'Bicyclist', -> rider 25
21: 'Motorcyclist', -> rider 25
22: 'Other Rider', -> rider 25
23: 'Lane Marking - Crosswalk', -> road 7
24: 'Lane Marking - General', -> road 7
25: 'Mountain', -> terrain 22
26: 'Sand', -> terrain 22
27: 'Sky', -> sky 23
28: 'Snow', -> terrain 22
29: 'Terrain', -> terrain 22
30: 'Vegetation', -> vegetation 21
31: 'Water', -> water 35
32: 'Banner', -> static 4
33: 'Bench', -> static 4
34: 'Bike Rack', -> static 4
35: 'Billboard', -> static 4
36: 'Catch Basin', -> static 4
37: 'CCTV Camera', -> static 4
38: 'Fire Hydrant', -> static 4
39: 'Junction Box', -> static 4
40: 'Mailbox', -> static 4
41: 'Manhole', -> static 4
42: 'Phone Booth', -> static 4
43: 'Pothole', -> static 4
44: 'Street Light', -> static 4
45: 'Pole', -> pole 17
46: 'Traffic Sign Frame', -> static 4
47: 'Utility Pole', -> pole 17
48: 'Traffic Light', -> traffic light 19
49: 'Traffic Sign (Back)', -> traffic sign 20
50: 'Traffic Sign (Front)', -> traffic sign 20
51: 'Trash Can', -> static 4
52: 'Bicycle', -> bicycle 33
53: 'Boat', -> dynamic 5
54: 'Bus', -> bus 28
55: 'Car', -> car 26
56: 'Caravan', -> caravan 29
57: 'Motorcycle', -> motorcycle 32
58: 'On Rails', -> train 31
59: 'Other Vehicle', -> dynamic 5
60: 'Trailer', -> trailer 30
61: 'Truck', -> truck 27
62: 'Wheeled Slow', -> dynamic 5
63: 'Car Mount', -> ego vehicle 1
64: 'Ego Vehicle', -> ego vehicle 1
65: 'Unlabeled', -> unlabeled 0
"""
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
import torch
from plyfile import PlyData
from os import path, listdir
from utils.cmaps import City36cmap, City19cmap
import json

class PDataset(Dataset):
    def __init__(self,
                 root_path="single_scenes/data",
                 scenes=['Town10HD_Opt_HardRainNight'],
                 sensors=['rgb', 'semantic', 'lidar'],
                 camera_positions=['D'],
                 lidar_positions=['T'],
                 extended_classes=False): # whether to use city19 or city36 class set
                 
        assert type(sensors) is list, "sensors must be a list"
        for s in sensors:
            assert s in ['rgb', 'depth', 'semantic', 'lidar'], "Unrecognized Sensor: "+s+", must be one of ['rgb', 'depth', 'semantic', 'lidar']"
        assert type(camera_positions) is list, "camera_positions must be a list"
        for p in camera_positions:
            assert p in ['F', 'FL', 'FR', 'D', 'L', 'R', 'B'], "Unrecognized Camera Position: "+p+", must be one of ['F', 'FL', 'FR', 'D', 'L', 'R', 'B']"
        assert type(lidar_positions) is list, "camera_positions must be a list"
        for p in lidar_positions:
            assert p in ['T', 'LL', 'LR'], "Unrecognized Camera Position: "+p+", must be one of ['T', 'LL', 'LR']"
            # positions relative to bbox center: T (-.65,   0, 1.7)
            #                                   LL ( 1.8, .85, .75) => T-LL (-2.45, -.85, .95)
            #                                   LR ( 1.8,-.85, .75) => T-RL (-2.45,  .85, .95)
        self.position_map = {'F': 'FRONT', 'FL': 'FRONT_LEFT',
                             'FR': 'FRONT_RIGHT', 'D': 'DESK',
                             'L': 'LEFT', 'R': 'RIGHT', 'B': 'BACK',
                             'T': 'TOP', 'LL': 'FRONT_LEFT', 'LR': 'FRONT_RIGHT'}
        self.sensor_map = {'rgb': 'CAM', 'depth': 'DEPTHCAM', 'semantic': 'SEGCAM', 'lidar': 'LIDAR'}
        self.sensors = sensors
        
        #scenes_path = [ for s in scenes]
        # change up so we don't need the open inside the comprehension
        # it is unknown wether is closes automatically
        self.sample_names = [(path.join(root_path, s), s+'_'+k) \
                                for s in scenes \
                                    for k in json.load(open(path.join(root_path, s, 'waypoints.json'), 'r'))]
        #self.sample_names = [(sp, f.split('.')[0]) for sp in scenes_path for f in listdir(path.join(sp, 'CAM_FRONT')) \
        #                        if path.isfile(path.join(sp, 'CAM_FRONT', f)) and not f.startswith('.')]
        
        self.folders = {}
        for s in self.sensors:
            self.folders[s] = {}
            if s != 'lidar':
                for p in camera_positions:
                    self.folders[s][p] = self.sensor_map[s]+'_'+self.position_map[p]
            else:
                for p in lidar_positions:
                    self.folders[s][p] = self.sensor_map[s]+'_'+self.position_map[p]
                    
                    
        if extended_classes:
            self.cmap = City36cmap
            self.raw2train = {0:0, 1:11, 2:13, 3:0, 4:0, 5:17, 6:7, 7:7, 8:8, 9:21, 10:1, 11:12, 12:20,
                              13:23, 14:6, 15:15, 16:10, 17:14, 18:19, 19:4, 20:5, 21:35, 22:22, 40:24,
                              41:25, 100:26, 101:27, 102:28, 103:31, 104:32, 105:33, 255:0}
            self.ignore_index = 0
        else:
            self.cmap = City19cmap
            self.raw2train = {-1:-1, 0:-1, 1:2, 2:4, 3:-1, 4:-1, 5:5, 6:0, 7:0, 8:1, 9:8, 10:-1,
                              11:3, 12:7, 13:10, 14:-1, 15:-1, 16:-1, 17:-1, 18:6, 19:-1, 20:-1,
                              21:-1, 22:9, 40:11, 41:12, 100:13, 101:14, 102:15, 103:16, 104:17,
                              105:18, 255:-1}
            self.ignore_index = -1
    
    def __getitem__(self, item):
        sp, fname = self.sample_names[item]
        out_dict = {}
        for s in self.sensors:
            out_dict[s] = {}
            for p in self.folders[s]:
                sample_path = path.join(sp, self.folders[s][p], fname)
                if s == 'rgb':
                    out_dict[s][p] = self.to_pytorch(self.load_rgb(sample_path+'.jpg'))
                elif s == 'depth':
                    out_dict[s][p] = self.load_depth(sample_path+'.png')
                elif s == 'semantic':
                    out_dict[s][p] = self.load_semantic(sample_path+'.png')
                else: # lidar
                    shift = 0. if p == 'T' else [-2.45, -.85, .95] if p == 'LL' else [-2.45, .85, .95]
                    out_dict[s][p] = self.load_lidar(sample_path+'.ply', xyz_shift=shift)
        return out_dict, fname
                    
    def __len__(self):
        return len(self.sample_names)
        
    @staticmethod
    def load_rgb(path):
        return cv.imread(path).copy()
        
    @staticmethod
    def to_pytorch(bgr):
        bgr = np.transpose(bgr.astype(np.float32)-[104.00698793, 116.66876762, 122.6789143], (2, 0, 1))
        return torch.from_numpy(bgr)
        
    def load_semantic(self, path):
        gt = cv.imread(path, cv.IMREAD_UNCHANGED)
        out = self.ignore_index*np.ones_like(gt, dtype=int)
        for k,v in self.raw2train.items():
            out[gt==k] = v
        return out
        
    @staticmethod
    def load_depth(path): # return the depth in meters
        t = cv.imread(path).astype(int)*np.array([256*256, 256, 1])
        return 1000.*t.sum(axis=2)/(256 * 256 * 256 - 1.)
        
    def load_lidar(self, path, xyz_shift=0.):
        data = PlyData.read(path)
        xyz = np.array([[x,y,z] for x,y,z,_,_ in data['vertex']])+xyz_shift
        l = np.array([l for _,_,_,_,l in data['vertex']])
        mapped = self.ignore_index*np.ones_like(l, dtype=int)
        for k,v in self.raw2train.items():
            mapped[l==k] = v
        return (xyz, mapped)
        
    def color_labels(self, label):
        assert len(label.shape) < 3, 'Input label must either be a grayscale image or a 1-dimensional array'
        return self.cmap[label]
        
"""
CityScapes 36 Label Set

unlabeled	            0
ego vehicle	            1
rectification border	2
out of roi	            3
static	                4
dynamic	                5
ground	                6
road	                7
sidewalk	            8
parking	                9
rail track	            10
building	            11
wall	                12
fence	                13
guard rail	            14
bridge	                15
tunnel	                16
pole	                17
polegroup	            18
t light	                19
t sign	                20
vegetation	            21
terrain	                22
sky	                    23
person	                24
rider	                25
car	                    26
truck	                27
bus	                    28
caravan	                29
trailer             	30
train	                31
motorcycle	            32
bicycle	                33
license plate	        34
water 	                35

"""
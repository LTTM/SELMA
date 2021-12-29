import numpy as np
import cv2 as cv
from datasets.dataset import BaseDataset
from utils.cmaps import City19cmap, Idd17cmap, Synthia16cmap, Idda16cmap, SII15cmap, Crosscity13cmap, CCI12cmap 
from utils.idmaps import city19_to_idd17, city19_to_synthia16, city19_to_idda16, city10_to_sii15, city19_to_crosscity13, city19_to_cci12
from utils.cnames import city19, idd17, synthia16, idda16, sii15, crosscity13, cci12

class CityDataset(BaseDataset):

    def __init__(self, class_set='city19', **kwargs):
        self.class_set = class_set
        
        super(CityDataset, self).__init__(**kwargs)

        self.init_idsmap()

    def init_ids(self):
        self.raw_to_train = {0:-1, 1:-1, 2:-1, 3:-1, 4:-1, 5:-1, 6:-1, 7:0, 8:1, 9:-1, 10:-1,
                             11:2, 12:3, 13:4, 14:-1, 15:-1, 16:-1, 17:5, 18:-1, 19:6, 20:7,
                             21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:-1, 30:-1,
                             31:16, 32:17, 33:18, 34:-1, 35:-1}
        self.ignore_index = -1
    
    def map_to_train(self, gt):
        gt_clone = self.ignore_index*np.ones(gt.shape, dtype=np.long)
        if self.raw_to_train is not None:
            for k,v in self.raw_to_train.items():
                gt_clone[gt==k] = v if self.to_cset is None else self.to_cset[v]
        return gt_clone
    
    def init_idsmap(self):
        if self.class_set == 'city19':
            self.to_cset = None
        elif self.class_set == 'idd17':
            self.to_cset = city19_to_idd17
        elif self.class_set == 'synthia16':
            self.to_cset = city19_to_synthia16
        elif self.class_set == 'idda16':
            self.to_cset = city19_to_idda16
        elif self.class_set == 'sii15':
            self.to_cset = city10_to_sii15
        elif self.class_set == 'crosscity13':
            self.to_cset = city19_to_crosscity13
        elif self.class_set == 'cci12':
            self.to_cset = city19_to_cci12
        
        
    def init_cmap(self):
        if self.class_set == 'city19':
            self.cmap = City19cmap
        elif self.class_set == 'idd17':
            self.cmap = Idd17cmap
        elif self.class_set == 'synthia16':
            self.cmap = Synthia16cmap
        elif self.class_set == 'idda16':
            self.cmap = Idda16cmap
        elif self.class_set == 'sii15':
            self.cmap = SII15cmap
        elif self.class_set == 'crosscity13':
            self.cmap = Crosscity13cmap
        elif self.class_set == 'cci12':
            self.cmap = CCI12cmap
        
    def init_cnames(self):
        if self.class_set == 'city19':
            self.cnames = city19
        elif self.class_set == 'idd17':
            self.cnames = idd17
        elif self.class_set == 'synthia16':
            self.cnames = synthia16
        elif self.class_set == 'idda16':
            self.cnames = idda16
        elif self.class_set == 'sii15':
            self.cnames = sii15
        elif self.class_set == 'crosscity13':
            self.cnames = crosscity13
        elif self.class_set == 'cci12':
            self.cnames = cci12
    
    @staticmethod
    def load_rgb(im_path):
        #image is read in bgr
        im = cv.imread(im_path)
        im = np.clip(np.round(im*255./np.array([245.,254.,240.])), a_min=0, a_max=255)
        return im.astype(np.uint8)
        
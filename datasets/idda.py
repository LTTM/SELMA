import numpy as np
from PIL import Image
from datasets.cityscapes import CityDataset

class IDDADataset(CityDataset):
    def __init__(self, class_set='idda16', **kwargs):
        super(IDDADataset, self).__init__(class_set=class_set, **kwargs)
        
    def init_ids(self):
        self.raw_to_train = {0:-1, 1:2, 2:4, 3:-1, 4:11, 5:5, 6:0, 7:0, 8:1, 9:8, 10:13,
                             11:3, 12:7, 13:6, 14:-1, 15:-1, 16:18, 17:17, 18:12, 19:9,
                             20:10, 21:-1, 22:-1, 23:-1}
        self.ignore_index = -1
from datasets.dataset import BaseDataset
from utils.cmaps import City19cmap

class CityDataset(BaseDataset):
    def init_ids(self):
        self.raw_to_train = {0:-1, 1:-1, 2:-1, 3:-1, 4:-1, 5:-1, 6:-1, 7:0, 8:1, 9:-1, 10:-1,
                             11:2, 12:3, 13:4, 14:-1, 15:-1, 16:-1, 17:5, 18:-1, 19:6, 20:7,
                             21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:-1, 30:-1,
                             31:16, 32:17, 33:18, 34:-1, 35:-1}
        self.ignore_index = -1

    def init_cmap(self):
        self.cmap = City19cmap
    
    def init_cnames(self):
        self.cnames = ["road",
                       "sidewalk",
                       "building",
                       "wall",
                       "fence",
                       "pole",
                       "traffic light",
                       "traffic sign",
                       "vegetation",
                       "terrain",
                       "sky",
                       "person",
                       "rider",
                       "car",
                       "truck",
                       "bus",
                       "train",
                       "motorbike",
                       "bycicle"]

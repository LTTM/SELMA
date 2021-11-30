import numpy as np
from PIL import Image
from datasets.cityscapes import CityDataset

class SYNTHIADataset(CityDataset):
    def init_ids(self):
        self.raw_to_train = {1:10, 2:2, 3:0, 4:1, 5:4, 6:8, 7:5, 8:13,
                             9:7, 10:11, 11:18, 12:17, 15:6, 16:9, 17:12,
                             18:14, 19:15, 20:16, 21:3}
        self.ignore_index = -1

    @staticmethod
    def load_semantic(path):
        gt_image = imageio.imread(path, format='PNG-FI')[:, :, 0]
        return np.uint8(gt_image)

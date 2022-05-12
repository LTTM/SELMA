from datasets.cityscapes import CityDataset

class IDDDataset(CityDataset):
    def __init__(self, class_set='idd17', **kwargs):
        super(IDDDataset, self).__init__(class_set=class_set, **kwargs)

    def init_ids(self):
        self.raw_to_train = {0:0, 1:-1, 2:0, 3:1, 4:-1, 5:-1, 6:11, 7:-1, 8:12, 9:17, 10:18, 11:-1,
                            12:13, 13:14, 14:15, 15:-1, 16:-1, 17:16, 18:-1, 19:-1, 20:3, 21:4,
                            22:-1, 23:-1, 24:7, 25:6, 26:5, 27:-1, 28:-1, 29:2, 30:-1, 31:-1,
                            32:8, 33:10, 34:-1, 251:-1, 252:-1, 253:-1, 254:-1, 255:-1}
        self.ignore_index = -1

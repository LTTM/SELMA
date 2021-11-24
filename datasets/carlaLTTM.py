from datasets.cityscapes import CityDataset

class LTTMDataset(CityDataset):
    def __init__(self,
                 town=None,
                 weather=None,
                 time_of_day=None,
                 **kwargs): # whether to use city19 or city36 class set

        super(LTTMDataset, self).__init__(**kwargs)

        if town is not None:
            self.items = [[town, e[1], e[2], e[3]] for e in self.items]

        if weather is not None:
            self.items = [[e[0], weather, e[2], e[3]] for e in self.items]

        if time_of_day is not None:
            self.items = [[e[0], e[1], time_of_day, e[3]] for e in self.items]

    def init_ids(self):
        self.raw_to_train = {-1:-1, 0:-1, 1:2, 2:4, 3:-1, 4:-1, 5:5, 6:0, 7:0, 8:1, 9:8, 10:-1,
                             11:3, 12:7, 13:10, 14:-1, 15:-1, 16:-1, 17:-1, 18:6, 19:-1, 20:-1,
                             21:-1, 22:9, 40:11, 41:12, 100:13, 101:14, 102:15, 103:16, 104:17,
                             105:18, 255:-1}
        self.ignore_index = -1

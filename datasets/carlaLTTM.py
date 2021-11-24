from datasets.cityscapes import CityDataset

class LTTMDataset(CityDataset):
    def __init__(self,
                 split_extension='csv',
                 split_separator=',',
                 split_skiplines=1,
                 town=None,
                 weather=None,
                 time_of_day=None,
                 **kwargs): # whether to use city19 or city36 class set

        super(LTTMDataset, self).__init__(split_extension=split_extension,
                                          split_separator=split_separator,
                                          split_skiplines=split_skiplines,
                                          **kwargs)

        if town is not None:
            self.items = [[town, e[1], e[2], e[3]] for e in self.items]

        if weather is not None:
            self.items = [[e[0], weather, e[2], e[3]] for e in self.items]

        if time_of_day is not None:
            self.items = [[e[0], e[1], time_of_day, e[3]] for e in self.items]

        self.towns_map    = {"01":        "Town01_Opt",
                             "02":        "Town02_Opt",
                             "03":        "Town03_Opt",
                             "04":        "Town04_Opt",
                             "05":        "Town05_Opt",
                             "06":        "Town06_Opt",
                             "07":        "Town07_Opt",
                             "10HD":      "Town10HD_Opt"}
        self.tods_map     = {"noon":      "Noon",
                             "night":     "Night",
                             "sunset":    "Sunset"}
        self.weathers_map = {"clear":     "Clear",
                             "wet":       "Wet",
                             "cloudy":    "Cloudy",
                             "wetcloudy": "WetCloudy",
                             "softrain":  "SoftRain",
                             "midrain":   "MidRain",
                             "hardrain":  "HardRain",
                             "midfog":    "MidFog",
                             "hardfog":   "HardFog"}

    def init_ids(self):
        self.raw_to_train = {-1:-1, 0:-1, 1:2, 2:4, 3:-1, 4:-1, 5:5, 6:0, 7:0, 8:1, 9:8, 10:-1,
                             11:3, 12:7, 13:10, 14:-1, 15:-1, 16:-1, 17:-1, 18:6, 19:-1, 20:-1,
                             21:-1, 22:9, 40:11, 41:12, 100:13, 101:14, 102:15, 103:16, 104:17,
                             105:18, 255:-1}
        self.ignore_index = -1

    def __getitem__(self, item):
        town, tod, weather, waypoint = self.items[item]
        folder = self.towns_map[town]+"_"+self.weathers_map[weather]+self.tods_map[tod]

        print(folder)

        # path =
        # rgb = self.load_rgb(path.join(self.root_path, rgb_path)) if if 'rgb' in self.sensors else None
        # gt = self.load_semantic(path.join(self.root_path, gt_path)) if 'semantic' in self.sensors else None
        #
        #
        # rgb, gt, _ = resize_and_crop(rgb=rgb, gt=gt)
        # rgb, gt, _ = data_augment(rgb=rgb, gt=gt)
        # rgb, gt, _ = to_pytorch(rgb=rgb, gt=gt)
        #
        # out_dict = {}
        # if rgb is not None: out_dict['rgb'] = rgb
        # if gt is not None: out_dict['semantic'] = gt
        # if depth is not None: out_dict['depth'] = depth
        #
        # return out_dict, item

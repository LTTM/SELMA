import json
from os import path
from plyfile import PlyData
from datasets.cityscapes import CityDataset

class LTTMDataset(CityDataset):
    def __init__(self,
                 split_extension='csv',
                 split_separator=',',
                 split_skiplines=1,
                 town=None,
                 weather=None,
                 time_of_day=None,
                 sensor_positions=["D"],
                 **kwargs): # whether to use city19 or city36 class set

        super(LTTMDataset, self).__init__(split_extension=split_extension,
                                          split_separator=split_separator,
                                          split_skiplines=split_skiplines,
                                          **kwargs)

        self.sensor_positions = sensor_positions

        if town is not None:
            self.items = [[town, e[1], e[2], e[3]] for e in self.items]

        if weather is not None:
            self.items = [[e[0], weather, e[2], e[3]] for e in self.items]

        if time_of_day is not None:
            self.items = [[e[0], e[1], time_of_day, e[3]] for e in self.items]

        self.towns_map      = {"01":        "Town01_Opt",
                               "02":        "Town02_Opt",
                               "03":        "Town03_Opt",
                               "04":        "Town04_Opt",
                               "05":        "Town05_Opt",
                               "06":        "Town06_Opt",
                               "07":        "Town07_Opt",
                               "10HD":      "Town10HD_Opt"}
        self.tods_map       = {"noon":      "Noon",
                               "night":     "Night",
                               "sunset":    "Sunset"}
        self.weathers_map   = {"clear":     "Clear",
                               "wet":       "Wet",
                               "cloudy":    "Cloudy",
                               "wetcloudy": "WetCloudy",
                               "softrain":  "SoftRain",
                               "midrain":   "MidRain",
                               "hardrain":  "HardRain",
                               "midfog":    "MidFog",
                               "hardfog":   "HardFog"}
        self.sensor_map     = {"rgb":       "CAM",
                               "semantic":  "SEGCAM",
                               "depth":     "DEPTHCAM",
                               "lidar":     "LIDAR",
                               "bbox":      "BBOX_LABELS"}
        self.file_ext       = {"rgb":       "jpg",
                               "semantic":  "png",
                               "depth":     "png",
                               "lidar":     "ply",
                               "bbox":      "json"}
        self.position_cam   = {"D":         "_DESK",
                               "F":         "_FRONT",
                               "FL":        "_FRONT_LEFT",
                               "FR":        "_FRONT_RIGHT",
                               "L":         "_LEFT",
                               "R":         "_RIGHT",
                               "B":         "_BACK"}
        self.position_lidar = {"T":         "_TOP",
                               "LL":        "_FRONT_LEFT",
                               "LR":        "_FRONT_RIGHT"}

    def init_ids(self):
        self.raw_to_train = {-1:-1, 0:-1, 1:2, 2:4, 3:-1, 4:-1, 5:5, 6:0, 7:0, 8:1, 9:8, 10:-1,
                             11:3, 12:7, 13:10, 14:-1, 15:-1, 16:-1, 17:-1, 18:6, 19:-1, 20:-1,
                             21:-1, 22:9, 40:11, 41:12, 100:13, 101:14, 102:15, 103:16, 104:17,
                             105:18, 255:-1}
        self.ignore_index = -1

    def __getitem__(self, item):
        town, tod, weather, waypoint = self.items[item]
        folder = self.towns_map[town]+"_"+self.weathers_map[weather]+self.tods_map[tod]
        wpath = path.join(self.root_path, folder, "%s%s", folder+"_"+waypoint+".%s")

        out_dict = {sensor:{} for sensor in sensors}

        for sensor in self.sensors:
            if sensor == "bbox":
                out_dict[sensor] = json.load(
                        open(wpath%(self.sensor_map[sensor], "", self.file_ext[sensor]), 'r')
                    )
            else:
                for position in self.sensor_positions:
                    if sensor == "lidar":
                        if position in self.position_lidar:
                            shift = 0. if p == 'T' else [-2.45, -.85, .95] if p == 'LL' else [-2.45, .85, .95]
                            out_dict[sensor][position] = self.load_lidar(
                                wpath%(self.sensor_map[sensor],
                                       self.position_lidar[position],
                                       self.file_ext[sensor]),
                                xyz_shift = shift
                            )
                    else:
                        if position in self.position_cam:
                            fun = self.load_rgb if sensor == "rgb" else self.load_semantic if sensor == "semantic" else self.load_depth
                            out_dict[sensor][position] = fun(
                                wpath%(self.sensor_map[sensor],
                                       self.position_cam[position],
                                       self.file_ext[sensor])
                                )

        poss = out_dict["rgb"].keys() if 'rgb' in out_dict else
                out_dict["semantic"].keys() if 'semantic' in out_dict else
                  out_dict["depth"].keys() if 'depth' in out_dict else []

        for pos in poss:
            rgb = out_dict['rgb'][pos] if 'rgb' in out_dict else None
            rgb = out_dict['gt'][pos] if 'gt' in out_dict else None
            rgb = out_dict['depth'][pos] if 'depth' in out_dict else None

            rgb, gt, depth = resize_and_crop(rgb=rgb, gt=gt, depth=depth)
            rgb, gt, depth = data_augment(rgb=rgb, gt=gt, depth=depth)
            rgb, gt, depth = to_pytorch(rgb=rgb, gt=gt, depth=depth)

            if rgb is not None: out_dict['rgb'][pos] = rgb
            if gt is not None: out_dict['gt'][pos] = rgb
            if depth is not None: out_dict['depth'][pos] = rgb

        return out_dict, item


def load_lidar(self, path, xyz_shift=0.):
    data = PlyData.read(path)
    xyz = np.array([[x,y,z] for x,y,z,_,_ in data['vertex']])+xyz_shift
    l = np.array([l for _,_,_,_,l in data['vertex']])
    mapped = self.ignore_index*np.ones_like(l, dtype=int)
    for k,v in self.raw_to_train.items():
        mapped[l==k] = v
    return (xyz, mapped)

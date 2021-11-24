import sys, os
sys.path.append(os.path.abspath('.'))

from datasets.carlaLTTM import LTTMDataset

t = LTTMDataset(root_path="datasets/demo",
                splits_path="lists",
                sensors=["rgb", "lidar", "semantic", "depth"],
                sensor_positions=["D","T"],
                town=None,
                weather=None,
                time_of_day=None,
                flip=True,
                gaussian_blur=True,
                blur_mul=5)

sample = t[0]

rgb = sample["rgb"]["D"]
semantic = sample["semantic"]["D"]
depth = sample["depth"]["D"]











#

import sys, os
sys.path.append(os.path.abspath('.'))

import numpy as np
from matplotlib import pyplot as plt

from datasets.carlaLTTM import LTTMDataset
from utils.bbox import *

t = LTTMDataset(root_path="datasets/demo",
                splits_path="lists",
                sensors=["rgb", "lidar", "semantic", "depth", "bbox"],
                sensor_positions=["D","T"],
                town=None,
                weather=None,
                time_of_day=None,
                flip=True,
                gaussian_blur=True,
                blur_mul=5)

data, ego, item = t[0]

rgb = t.to_rgb(data["rgb"]["D"])
semantic = t.color_label(data["semantic"]["D"])
depth = np.log(data["depth"]["D"])
pc_geom, pc_labels = data["lidar"]["T"]
bbox = data["bbox"]

bbox_centers = np.array([
                [ e['location']['x'],
                  e['location']['y'],
                  e['location']['z'],
                ]
               for e in bbox ])
ego_pos = np.array([ego["Location"]["x"], ego["Location"]["y"], ego["Location"]["z"]])
bbox_centers -= ego_pos

ego_inverse = make_rotation_matrix(**ego["Rotation"])
pc_geom = np.matmul(ego_inverse, pc_geom.T).T
lidar_rot = make_rotation_matrix(yaw=-90)
#pc_geom -= np.array([-0.65, 0, 1.7])
pc_geom = np.matmul(lidar_rot, pc_geom.T).T

fig, axs = plt.subplots(2,2)
axs[0,0].imshow(rgb)
axs[0,1].imshow(semantic)
axs[1,0].imshow(depth)
axs[1,1].scatter(pc_geom[:,0], pc_geom[:,1], s=1)
axs[1,1].scatter(bbox_centers[:,0], bbox_centers[:,1], s=1)

plt.show()

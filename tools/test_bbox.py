from utils.bbox import project_boxes
from utils.cmaps import City36cmap
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import json
from os import path

rootdir = 'Z:/datasets/PDataset/Town10HD_Opt_ClearNoon'
fname = "Town10HD_Opt_ClearNoon_101228321417574247"
sensor = 'CAM_FRONT'
id_map = {0:0, 1:11, 2:13, 3:0, 4:0, 5:17, 6:7, 7:7, 8:8, 9:21, 10:1, 11:12, 12:20,
          13:23, 14:6, 15:15, 16:10, 17:14, 18:19, 19:4, 20:5, 21:35, 22:22, 40:24,
          41:25, 100:26, 101:27, 102:28, 103:31, 104:32, 105:33, 255:0}

with open(path.join(rootdir, 'calibrated_sensor.json'), 'r') as f:
    sensors = json.load(f)
snames = [s['sensor_name'] for s in sensors]
sidx = snames.index(sensor)
sdata = sensors[sidx]

with open(path.join(rootdir, 'waypoints.json'), 'r') as f:
    waypoints = json.load(f)
wdata = waypoints[fname.split('_')[-1]]

with open(path.join(rootdir, 'BBOX_LABELS', fname+'.json'), 'r') as f:
    bboxes = json.load(f)

t, im_w, im_h = project_boxes(bboxes, wdata, sdata, id_map)

im = cv.imread(path.join(rootdir, sensor, fname+'.jpg'))

fig, ax = plt.subplots(1,1)
ax.imshow(im[...,::-1])
for corners, label in t:
    ax.plot(corners[:,0], corners[:,1], color=City36cmap[label]/255.)
ax.set_xlim([0, im_w])
ax.set_ylim([0, im_h])
ax.invert_yaxis()
plt.show()
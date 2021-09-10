import glob
import os
import sys

try:
    cpath = 'carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')
    sys.path.append(glob.glob(cpath)[0])
except IndexError:
    raise ValueError("carla not found", cpath)

import carla
import time

client = carla.Client('127.0.0.1', 2000)
world = client.get_world()
#world = client.load_world('Town01')
#time.sleep(5)
#t = world.get_random_location_from_navigation()
t = client.get_available_maps()
print(t)
import glob
import os
import sys

try:
    cpath = '../PreCompiled/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')
    sys.path.append(glob.glob(cpath)[0])
except IndexError:
    raise ValueError("carla not found", cpath)

import carla

import random
import time
from custom_bp_tags import override_parked_vehicles
from itertools import chain

client = carla.Client('127.0.0.1', 2000)
world = client.reload_world()

override_parked_vehicles(world)
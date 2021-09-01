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
from custom_bp_tags import custom_tag
from itertools import chain

client = carla.Client('127.0.0.1', 2000)
world = client.reload_world()
#world = client.get_world()

blueprint_library = world.get_blueprint_library()
walkers = blueprint_library.filter('walker')
vehicles = blueprint_library.filter('vehicle')
#bikes = [x for x in vehicles if int(x.get_attribute('number_of_wheels')) == 2]
s = world.get_spectator()
t0 = carla.Transform(carla.Location(0, 10, 502), carla.Rotation(0, -90, 0))
s.set_transform(t0)

#"""
sensor = blueprint_library.find('sensor.camera.semantic_segmentation')
sensor.set_attribute('image_size_x', '1920')
sensor.set_attribute('image_size_y', '1080')
sensor.set_attribute('fov', '110')
"""
sensor = blueprint_library.find('sensor.lidar.ray_cast_semantic')
sensor.set_attribute('channels',str(256))
sensor.set_attribute('points_per_second',str(1000000))
sensor.set_attribute('rotation_frequency',str(64))
sensor.set_attribute('range', str(50))
"""
sensor.set_attribute('sensor_tick', '.1')

c = world.spawn_actor(sensor, t0)
c.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame))
#c.listen(lambda point_cloud: point_cloud.save_to_disk('output/%06d.ply' % point_cloud.frame))

#"""
spawned = []
for i, bp in enumerate(chain(walkers, vehicles)):
    tr = carla.Transform(carla.Location(10*i, 0, 500), carla.Rotation(0, 90, 0))
    tags_dict = custom_tag(bp)
    print(str(i)+'\t'+bp.id+'\t', tags_dict)
    v = world.spawn_actor(bp, tr)
    if tags_dict is not None:
        v.update_semantic_tags(tags_dict)
    v.set_simulate_physics(False)
    spawned.append(v)
#"""

for j in range(10*i):
    time.sleep(.1)
    t0.location.x += 1
    s.set_transform(t0)
    c.set_transform(t0)
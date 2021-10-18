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
import numpy as np
import random

client = carla.Client('127.0.0.1', 2000)
world = client.get_world()                            
                                 
ws = {'Default': carla.WeatherParameters.Default,
      'ClearNoon': carla.WeatherParameters.ClearNoon,
      'CloudyNoon': carla.WeatherParameters.CloudyNoon,
      'WetNoon': carla.WeatherParameters.WetNoon,
      'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
      'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
      'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
      'ClearSunset': carla.WeatherParameters.ClearSunset,
      'CloudySunset': carla.WeatherParameters.CloudySunset,
      'WetSunset': carla.WeatherParameters.WetSunset,
      'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,
      'MidRainSunset': carla.WeatherParameters.MidRainSunset,
      'HardRainSunset': carla.WeatherParameters.HardRainSunset,
      'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
      'ClearNight': carla.WeatherParameters.ClearNight,
      'CloudyNight': carla.WeatherParameters.CloudyNight,
      'WetNight': carla.WeatherParameters.WetNight,
      'WetCloudyNight': carla.WeatherParameters.WetCloudyNight,
      'SoftRainNight': carla.WeatherParameters.SoftRainNight,
      'MidRainyNight': carla.WeatherParameters.MidRainyNight,
      'HardRainNight': carla.WeatherParameters.HardRainNight,
      'MidFoggyNoon': carla.WeatherParameters.MidFoggyNoon,
      'HardFoggyNoon': carla.WeatherParameters.HardFoggyNoon,
      'MidFoggySunset': carla.WeatherParameters.MidFoggySunset,
      'HardFoggySunset': carla.WeatherParameters.HardFoggySunset,
      'MidFoggyNight': carla.WeatherParameters.MidFoggyNight,
      'HardFoggyNight': carla.WeatherParameters.HardFoggyNight}

s = world.get_spectator()
s.set_transform(carla.Transform(carla.Location(x=101, y=-27, z=3), carla.Rotation(pitch=-19, yaw=108, roll=0)))
bp = world.get_blueprint_library().filter('sensor.camera.rgb')[0]
bp.set_attribute('image_size_x', '1920')
bp.set_attribute('image_size_y', '1080')
c = world.spawn_actor(bp, s.get_transform())

w = 'tmp'
skip = True

def save(image):
    global w
    global skip
    if not skip:
        image.save_to_disk('F:/Carla/weathers/'+w+'.png')

c.listen(save)

for w in ws:
    skip = True
    print(w)
    world.set_weather(ws[w])
    time.sleep(5)
    skip = False
    time.sleep(2)
    skip = True
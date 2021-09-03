import glob
import os
import sys

try:
    cpath = '../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')
    sys.path.append(glob.glob(cpath)[0])
except IndexError:
    raise ValueError("carla not found", cpath)

import carla
import random


cars =   [
            'vehicle.audi.a2',
            'vehicle.nissan.micra',
            'vehicle.audi.tt',
            'vehicle.mercedes.coupe_2020',
            'vehicle.bmw.grandtourer',
            'vehicle.micro.microlino',
            'vehicle.ford.mustang',
            'vehicle.chevrolet.impala',
            'vehicle.lincoln.mkz_2020',
            'vehicle.citroen.c3',
            'vehicle.dodge.charger_police',
            'vehicle.nissan.patrol',
            'vehicle.jeep.wrangler_rubicon',
            'vehicle.mini.cooper_s',
            'vehicle.mercedes.coupe',
            'vehicle.dodge.charger_2020',
            'vehicle.seat.leon',
            'vehicle.toyota.prius',
            'vehicle.tesla.model3',
            'vehicle.audi.etron',
            'vehicle.lincoln.mkz_2017',
            'vehicle.dodge.charger_police_2020',
            'vehicle.mini.cooper_s_2021',
            'vehicle.nissan.patrol_2021'
         ]
trucks = [
            'vehicle.ford.ambulance',
            'vehicle.carlamotors.firetruck',
            'vehicle.carlamotors.carlacola',
            'vehicle.tesla.cybertruck',
            'vehicle.mercedes.sprinter',
            'vehicle.volkswagen.t2',
            'vehicle.lttm.truck02',
            'vehicle.lttm.truck04'
         ]
busses = [
            'vehicle.lttm.bus01',
            'vehicle.lttm.bus02'
         ]
trains = [
            'vehicle.lttm.train01',
            'vehicle.lttm.train02'
         ]
mbikes = [
            'vehicle.harley-davidson.low_rider',
            'vehicle.yamaha.yzf',
            'vehicle.kawasaki.ninja',
            'vehicle.vespa.zx125'
          ]
bikes =  [
            'vehicle.bh.crossbike',
            'vehicle.gazelle.omafiets',
            'vehicle.diamondback.century'
         ]


joined = cars+busses+trucks+mbikes+bikes


def custom_tag(bp):
    # extract blueprint id (string)
    bp_id = bp.id
    
    # the only blueprints that require attention 
    # are the vehicle and walker ones
    if bp_id.startswith('walker'):
        return {4: 40} # pedestrian->person
    if bp_id in cars:
        return {4:41, 10:100} # pedestrian->rider, vehicle->car
    if bp_id in trucks:
        return {0:101, 4:41, 10:101} # pedestrian->rider, vehicle->truck
    if bp_id in busses:
        return {0:102, 4:41, 10:102} # pedestrian->rider, vehicle->bus
    if bp_id in trains:
        return {0:103, 4:41, 10:103} # pedestrian->rider, vehicle->train
    if bp_id in mbikes:
        return {4:41, 10:104} # pedestrian->rider, vehicle->motorcycles
    if bp_id in bikes:
        return {4:41, 10:105} # pedestrian->rider, vehicle->bicycle

    return None

def override_parked_vehicles(world):
    parked_objs = world.get_environment_objects(carla.CityObjectLabel.Vehicles)
    trs = [carla.Transform(obj.bounding_box.location, obj.bounding_box.rotation) for obj in parked_objs]
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    blueprint_library = world.get_blueprint_library()
    for tr in trs:
        tr.location.z += .5
        bp = blueprint_library.filter(random.choice(joined))[0]
        tags_dict = custom_tag(bp)
        try:
            v = world.spawn_actor(bp, tr)
            if tags_dict is not None:
                v.update_semantic_tags(tags_dict)
        except RuntimeError:
            pass
    
"""
Classes:

    None         =   0u,
    Buildings    =   1u,
    Fences       =   2u,
    Other        =   3u,
    Pedestrians  =   4u,
    Poles        =   5u,
    RoadLines    =   6u,
    Roads        =   7u,
    Sidewalks    =   8u,
    Vegetation   =   9u,
    Vehicles     =  10u,
    Walls        =  11u,
    TrafficSigns =  12u,
    Sky          =  13u,
    Ground       =  14u,
    Bridge       =  15u,
    RailTrack    =  16u,
    GuardRail    =  17u,
    TrafficLight =  18u,
    Static       =  19u,
    Dynamic      =  20u,
    Water        =  21u,
    Terrain      =  22u,
    Persons      =  40u,
    Riders       =  41u,
    Cars         =  100u,
    Trucks       =  101u,
    Busses       =  102u,
    Trains       =  103u,
    Motorcycles  =  104u,
    Bycicles     =  105u,
    Any          =  0xFF

"""
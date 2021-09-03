import numpy as np

class cityCMAP():
    def __init__(self):
        self.cmap = np.zeros((256, 3), dtype=np.uint8)
        self.cmap[7] = [128, 64, 128]    # road
        self.cmap[8] = [244, 35, 232]    # sidewalk
        self.cmap[1] = [70, 70, 70]      # Building
        self.cmap[11] = [102, 102, 156]  # Wall
        self.cmap[2] = [190, 153, 153]   # Fence
        self.cmap[5] = [153, 153, 153]   # Pole
        self.cmap[18] = [250, 170, 30]   # Traffic Light
        self.cmap[12] = [220, 220, 0]    # Traffic Sign
        self.cmap[9] = [107, 142, 35]    # Vegetation
        self.cmap[22] = [152, 251, 152]  # Terrain
        self.cmap[13] = [0, 130, 180]    # Sky
        self.cmap[40] = [220, 20, 60]    # Person
        self.cmap[41] = [255, 0, 0]      # Rider
        self.cmap[100] = [0, 0, 142]     # Car
        self.cmap[101] = [0, 0, 70]      # Truck
        self.cmap[102] = [0, 60, 100]    # Bus
        self.cmap[103] = [0, 80, 100]    # Train
        self.cmap[104] = [0, 0, 230]     # Motorbike
        self.cmap[105] = [119, 11, 32]   # Bicycle
    
    def color(self, im):
        assert len(im.shape) == 2, "Input image must be grayscale"
        out = self.cmap[im]
        return out
    
    
        
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
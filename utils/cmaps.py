import numpy as np

#######################################
# ------ CITYSCAPES 36 CLASSES ------ #
#######################################

City36cmap = np.zeros((256,3), dtype=np.uint8)  # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
City36cmap[4:34,...] = np.array([[110,190,160], # static
                                 [111, 74,  0], # dynamic
                                 [ 81, 0,  81], # ground
                                 [128, 64,128], # road
                                 [244, 35,232], # sidewalk
                                 [250,170,160], # parking
                                 [230,150,140], # rail track
                                 [ 70, 70, 70], # building
                                 [102,102,156], # wall
                                 [190,153,153], # fence
                                 [180,165,180], # guard rail
                                 [150,100,100], # bridge
                                 [150,120, 90], # tunnel
                                 [153,153,153], # pole
                                 [153,153,153], # polegroup
                                 [250,170, 30], # traffic light
                                 [220,220,  0], # traffic sign
                                 [107,142, 35], # vegetation
                                 [152,251,152], # terrain
                                 [ 70,130,180], # sky
                                 [220, 20, 60], # person
                                 [255,  0,  0], # rider
                                 [  0,  0,142], # car
                                 [  0,  0, 70], # truck
                                 [  0, 60,100], # bus
                                 [  0,  0, 90], # caravan
                                 [  0,  0,110], # trailer
                                 [  0, 80,100], # train
                                 [  0,  0,230], # motorcycle
                                 [119, 11, 32]])# bicycle
City36cmap[25,...] =             [45, 60, 150]  # water
City36cmap[-1,...] =             [  0,  0,142]  # licence plate


#######################################
# ------ CITYSCAPES 19 CLASSES ------ #
#######################################

City19cmap = np.array([[128, 64,128], # road
                       [244, 35,232], # sidewalk
                       [ 70, 70, 70], # building
                       [102,102,156], # wall
                       [190,153,153], # fence
                       [153,153,153], # pole
                       [250,170, 30], # traffic light
                       [220,220,  0], # traffic sign
                       [107,142, 35], # vegetation
                       [152,251,152], # terrain
                       [ 70,130,180], # sky
                       [220, 20, 60], # person
                       [255,  0,  0], # rider
                       [  0,  0,142], # car
                       [  0,  0, 70], # truck
                       [  0, 60,100], # bus
                       [  0, 80,100], # train
                       [  0,  0,230], # motorcycle
                       [119, 11, 32], # bicycle
                       [  0,  0,  0]], dtype=np.uint8) # void class


#######################################
# -------- SYNTHIA16 CLASSES -------- #
#######################################

City16cmap = np.array([[128, 64,128], # road
                       [244, 35,232], # sidewalk
                       [ 70, 70, 70], # building
                       [250,170, 30], # traffic light
                       [220,220,  0], # traffic sign
                       [107,142, 35], # vegetation
                       [152,251,152], # terrain
                       [ 70,130,180], # sky
                       [220, 20, 60], # person
                       [255,  0,  0], # rider
                       [  0,  0,142], # car
                       [  0,  0, 70], # truck
                       [  0, 60,100], # bus
                       [  0, 80,100], # train
                       [  0,  0,230], # motorcycle
                       [119, 11, 32], # bicycle
                       [  0,  0,  0]], dtype=np.uint8) # void class


#######################################
# --------- IDDA 16 CLASSES --------- #
#######################################

Idda16cmap = np.array([[128, 64,128], # road
                       [244, 35,232], # sidewalk
                       [ 70, 70, 70], # building
                       [102,102,156], # wall
                       [190,153,153], # fence
                       [153,153,153], # pole
                       [250,170, 30], # traffic light
                       [220,220,  0], # traffic sign
                       [107,142, 35], # vegetation
                       [152,251,152], # terrain
                       [ 70,130,180], # sky
                       [220, 20, 60], # person
                       [255,  0,  0], # rider
                       [  0,  0,142], # car
                       [  0,  0,230], # motorcycle
                       [119, 11, 32], # bicycle
                       [  0,  0,  0]], dtype=np.uint8) # void class
                       

#######################################
# ------ IDDASYNTHIA15 CLASSES ------ #
#######################################

IddaSynth15cmap = np.array([[128, 64,128], # road
                            [244, 35,232], # sidewalk
                            [ 70, 70, 70], # building
                            [102,102,156], # wall
                            [190,153,153], # fence
                            [153,153,153], # pole
                            [250,170, 30], # traffic light
                            [220,220,  0], # traffic sign
                            [107,142, 35], # vegetation
                            [ 70,130,180], # sky
                            [220, 20, 60], # person
                            [255,  0,  0], # rider
                            [  0,  0,142], # car
                            [  0,  0,230], # motorcycle
                            [119, 11, 32], # bicycle
                            [  0,  0,  0]], dtype=np.uint8) # void class


#######################################
# ------- CROSSCITY13 CLASSES ------- #
#######################################

City13cmap = np.array([[128, 64,128], # road
                       [244, 35,232], # sidewalk
                       [ 70, 70, 70], # building
                       [250,170, 30], # traffic light
                       [220,220,  0], # traffic sign
                       [107,142, 35], # vegetation
                       [ 70,130,180], # sky
                       [220, 20, 60], # person
                       [255,  0,  0], # rider
                       [  0,  0,142], # car
                       [  0, 60,100], # bus
                       [  0,  0,230], # motorcycle
                       [119, 11, 32], # bicycle
                       [  0,  0,  0]], dtype=np.uint8) # void class
                       

#######################################
# ----- CROSSCITYIDDA12 CLASSES ----- #
#######################################

CrossCity13cmap = np.array([[128, 64,128], # road
                            [244, 35,232], # sidewalk
                            [ 70, 70, 70], # building
                            [250,170, 30], # traffic light
                            [220,220,  0], # traffic sign
                            [107,142, 35], # vegetation
                            [ 70,130,180], # sky
                            [220, 20, 60], # person
                            [255,  0,  0], # rider
                            [  0,  0,142], # car
                            [  0,  0,230], # motorcycle
                            [119, 11, 32], # bicycle
                            [  0,  0,  0]], dtype=np.uint8) # void class
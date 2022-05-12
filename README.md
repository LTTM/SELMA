# SELMA
Thank you for considering the SELMA dataset for your work.

**SELMA v0.1 has just been released**. 
SELMA is a novel synthetic dataset for semantic segmentation based on the [CARLA](https://carla.org/) simulator, that contains more than 30K unique waypoints acquired from 24 different sensors including RGB, depth, semantic cameras and LiDARs, in
27 different atmospheric and daytime conditions, for a total of more than
20M samples and 8 TB of (compressed) data.

## Dataset Information
All the information about SELMA is provided in [our paper](https://arxiv.org/abs/2204.09788). If you use SELMA, please cite it in your work.

Here, we report some techical details to help quickstarting your work with SELMA.
### Data Description
SELMA was carefully designed to address the need of the autonomous driving research community, by encompassing a wide diversity of weathers, environments, sensors, and times-of-day.
Specifically:
- Data from **8 towns** (Town01_Opt-Town07_Opt, Town10HD_Opt) is available. A detailed description of the towns is available at https://carla.readthedocs.io/en/0.9.12/core_map/#carla-maps.
- **24 sensors** were used:
  - 3 Semantic LiDARs (TOP\LEFT\RIGHT): beside the standard spatial coordinates, the pointcloud embeds the semantic label for each point.
  - 7 RGB/Depth/Semantic Cameras (DESK\FRONT\FRONTRIGHT\FRONTLEFT\RIGHT\LEFT\BACK): high-quality 1280x640 images, with photo-realistic effects. Semantic Cameras are used as semantic groundtruth. The RGB images are provided in JPEG, the depth and semantic ones in PNG. The three type of cameras are co-located at each position.\
All sensors are synchronized to acquire data simultanously.
- The same acquisition is available under **9 different weathers** and **3 different times-of-day**, for a total of **27 unique environmental conditions**. Note that the scenes change across different weather conditions: the same waypoint will generate a different scene when changing weather condition, e.g., the actors are re-spawned every time at random.
### Data Format
The acquisitions are archived per *scene* and per *sensor*. A scene is a combination of a town and an environmental condition (weather and time-of-day). For instance, *Town02_Opt_CloudyNoon* is the scene acquired in *Town02_Opt* with a *Cloudy* sky at *Noon*. From the website it is possible to select the data to download by combining different towns and environmental conditions. Furthermore, we offer the possibility to download only a subset of sensor data (e.g., only that acquired by the *CAM_LEFT*, i.e., the camera on the left side of the vehicle).

The data will be downloaded in a subfolder structure `<SCENE_NAME>/<SENSOR_NAME>`\ where `<SCENE_NAME>` is obtained as `<Town_EnvironmentalCondition>`.\
The list of towns and environmental conditions is reported in the following tables:

|                |          |          | SCENES   |          |          |          |          |
|----------------|----------|----------|----------|----------|----------|----------|----------|
|**Weathers**    |Clear     |Cloudy    |Wet       |WetCloudy |SoftRain  |MidRainy  |HardRain  |
|**Times-of-day**|Noon      |Sunset    |Night     |
|**Towns**       |Town01_Opt|Town02_Opt|Town03_Opt|Town04_Opt|Town05_Opt|Town06_Opt|Town07_Opt|Town10HD_Opt|

### Semantic Classes
Semantic labeling for both camera and LiDAR data into 36 distinct classes, with complete overlap with the training set of common benchmarks like [Cityscapes](https://www.cityscapes-dataset.com/), obtained by modifying the source code of the CARLA.
We will provide soon a dataloader to remap the SELMA classes to the most popular benchmarks.
For now, the following mapping can be used to map from 36 classes to 19 classes (Cityscapes style):

```raw_to_train = {-1:-1, 0:-1, 1:2, 2:4, 3:-1, 4:-1, 5:5, 6:0, 7:0, 8:1, 9:8, 11:3, 12:7, 13:10, 14:-1, 15:-1, 16:-1, 17:-1, 18:6, 19:-1, 21:-1, 22:9, 40:11, 41:12, 100:13, 101:14, 102:15, 103:16, 105:18, 255:-1}```

Where -1 indicates the void classes.

|      |          |          |      |       | LABELS   |             |            |          |       |
|:----:|:--------:|:--------:|:----:|:-----:|:----:|:-------------:|:------------:|:----------:|:-------:|
|road|sidewalk|building|wall|fence|pole|traffic light|traffic sign|vegetation|terrain|
|sky |person  |rider   |car |truck|bus |train        |motorbike   |bike      |       |



## Download Instructions
1. Navigate to https://scanlab.dei.unipd.it/app/dataset
2. Login or register to access the data. We need you to register to monitor the traffic to/from our server and avoid overloading it. We will only use your e-mail for very limited service communications (new dataset releases, server status).
3. Select the subset of data you want to download using the selection tool available on the dataset page. We suggest to start with the SELMA dataset (i.e., the random SELMA split).
4. When prompted, download and unzip the archive. The zip contains the downloader toolkit: an executable downloader and a json file with the list of files to download.
5. Enable the execution of the downloader for your platform changing the permission accordingly (`chmod +x downloader-`<*yourOS*\>`.sh` on Unix systems, double click and accept on Windows).
6. Execute the downloader and.. you're set! You will find a new `CV` folder in the `dataset-downloader-toolkit` folder with all the data. 

## Cite us
If you find this work usefule, please consider to cite us

       @inProceedings{testolina2022SELMA,
       title={SELMA: SEmantic Large-scale Multimodal Acquisitions in Variable Weather, Daytime and Viewpoints},
       author={Testolina, Paolo and Barbato, Francesco and Michieli, Umberto and Giordani, Marco and Zanuttigh, Pietro and Zorzi, Michele},
       journal={arXiv preprint arXiv:2204.09788},
       year={2022}
       }

## Additional Information and Support
- Notice that the download may take considerable time, depending on how much data you selected
- We suggest to start with the SELMA rand split, to get an overview of the dataset, and then select all the data you want.
- Right now, you can download SELMA v0.1. It is the very first release, and our first attempt at building something so colossal. Do not hesitate to contact us through the website [Contact Information module](https://scanlab.dei.unipd.it/) or at testolina@dei.unipd.it for information, troubleshooting and support.

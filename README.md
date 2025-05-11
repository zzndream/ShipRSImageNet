**ShipRSImagaeNet**: A Large-scale Fine-Grained Dataset for Ship Detection in High-Resolution Optical Remote Sensing Images
========
[![python](https://img.shields.io/badge/Python-3.x-ff69b4.svg)](https://github.com/luyanger1799/Amazing-Semantic-Segmentation.git)
[![OpenCV](https://img.shields.io/badge/OpenCV-3.x%7C4.x-orange.svg)](https://github.com/luyanger1799/Amazing-Semantic-Segmentation.git)
[![Apache](https://img.shields.io/badge/Apache-2.0-blue.svg)](https://github.com/luyanger1799/Amazing-Semantic-Segmentation.git)

# Description

<font color=red>ShipRSImageNet</font> is a large-scale fine-grainted dataset for ship detection in high-resolution optical remote sensing images. The dataset contains <font color=red>3,435 images</font> from various sensors, satellite platforms, locations, and seasons. Each image is around 930×930 pixels and contains ships with different scales, orientations, and aspect ratios. The images are annotated by experts in satellite image interpretation, categorized into <font color=red>50 object categories images</font>. The fully annotated ShipRSImageNet contains <font color=red>17,573 ship instances</font>. There are five critical contributions of the proposed ShipRSImageNet dataset compared with other existing remote sensing image datasets.

- ***Images are collected from various remote sensors cover- ing multiple ports worldwide and have large variations in size, spatial resolution, image quality, orientation, and environment.***

- ***Ships are hierarchically classified into four levels and 50 ship categories.***

- ***The number of images, ship instances, and ship cate- gories is larger than that in other publicly available ship datasets. Besides, the number is still increasing.***

- ***We simultaneously use both horizontal and oriented bounding boxes, and polygons to annotate images, providing detailed information about direction, background, sea environment, and location of targets.***

-  ***We have benchmarked several state-of-the-art object detection algorithms on ShipRSImageNet, which can be used as a baseline for future ship detection methods.***

# What's New

**Test set** has been released to make the dataset more comprehensive and accessible. In ShipRSImageNet V1.1, we are excited to announce the addition of a dedicated **test set**. This update significantly enhances the dataset by providing a standardized evaluation split, enabling fair comparison and benchmarking of ship detection algorithms. The new test set is carefully curated to reflect diverse scenarios and ship categories, ensuring robust assessment of model performance in real-world conditions. We hope this improvement will facilitate further research and innovation in fine-grained ship detection and classification using high-resolution remote sensing imagery.

# Examples of Annotated Images

![image](https://github.com/zzndream/ShipRSImageNet/blob/main/imgs/Examples%20of%20Annotated%20Images.jpeg)

# Image Source and Usage License

The ShipRSImageNet dataset collects images from a variety of sensor platforms and datasets, in particular:

- Images of the xView dataset are collected from WorldView-3 satellites with 0.3m ground resolution. Images in xView are pulled from a wide range of geographic locations. We only extract images with ship targets from them. Since the image in xView is huge for training, we slice them into 930×930 pixels with 150 pixels overlap to produce 532 images and relabeled them with both horizontal bounding box and oriented bounding box.

- We also collect 1,057 images from HRSC2016 and 1,846 images from FGSD datasets, corrected the mislabeled and relabeled missed small ship targets.

- 21 images from the Airbus Ship Detection Challenge.

- 17 images from Chinese satellites suchas GaoFen-2 and JiLin-1.

Use of the Google Earth images must respect the ["Google Earth" terms of use](https://www.google.com/permissions/geoguidelines.html).

All images and their associated annotations in ShipRSImageNet **can be used for academic purposes only, but any commercial use is prohibited**.

# Object Category

The ship classification tree of proposed ShipRSImageNet is shown in the following figure. Level 0 distinguish whether the object is a ship, namely Class. Level 1 further classifies the ship object category, named as Category. Level 2 further subdivides the categories based on Level 1. Level 3 is the specific type of ship, named as Type. 

![image](https://github.com/zzndream/ShipRSImageNet/blob/main/imgs/ShipRSImageNet_categories_tree.jpeg)

At Level 3, ship objects are divided into **50** types. For brevity, we use the following abbreviations: DD for Destroyer, FF for Frigate, LL for Landing, AS for Auxiliary Ship, LSD for Landing Ship Dock, LHA for Landing Heli- copter Assault Ship, AOE for Fast Combat Support Ship, EPF for Expeditionary Fast Transport Ship, and RoRo for Roll- on Roll-off Ship. These 50 object classes are Other Ship, Other Warship, Submarine, Other Aircraft Carrier, Enterprise, Nimitz, Midway, Ticonderoga, Other Destroyer, Atago DD, Arleigh Burke DD, Hatsuyuki DD, Hyuga DD, Asagiri DD, Other Frigate, Perry FF, Patrol, Other Landing, YuTing LL, YuDeng LL, YuDao LL, YuZhao LL, Austin LL, Osumi LL, Wasp LL, LSD 41 LL, LHA LL, Commander, Other Auxiliary Ship, Medical Ship, Test Ship, Training Ship, AOE, Masyuu AS, Sanantonio AS, EPF, Other Merchant, Container Ship, RoRo, Cargo, Barge, Tugboat, Ferry, Yacht, Sailboat, Fishing Vessel, Oil Tanker, Hovercraft, Motorboat, and Dock.

# Dataset Download

- Baidu Drive (Extraction code:qd7e):
  - [ShipRSImageNet V1.1](https://pan.baidu.com/s/1-s39q906ROYxNiZgvjYKzQ](https://pan.baidu.com/s/1-s39q906ROYxNiZgvjYKzQ?pwd=qd7e)


# Benchmark Code Installation

We keep all the experiment settings and hyper-parameters the same as depicted in MMDetection(v2.11.0) config files except for the number of categories and parameters. MMDe- tection is an open-source object detection toolbox based on PyTorch. It is a part of the OpenMMLab project developed by Multimedia Laboratory, CUHK.

This project is based on [MMdetection](https://github.com/open-mmlab/mmdetection)(v2.11.0). MMDetection is an open source object detection toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project. 

## Prerequisites

- Linux or macOS (Windows is in experimental support)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

## Installation

- Install MMdetection following [the instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md). We are noting that our code is checked in mmdetection V2.11.0 and pytorch V1.7.1.

  - Create a conda virtual environment and activate it. 

    ```python
    conda create -n open-mmlab python=3.7 -y
    conda activate open-mmlab
    ```

  - Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```python
    conda install pytorch torchvision -c pytorch
    ```

    Note: Make sure that your compilation CUDA version and runtime CUDA version match. You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

  - Install mmcv-full, we recommend you to install the pre-build package as below.

    ```python
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    ```

    Please replace `{cu_version}` and `{torch_version}` in the url to your desired one. For example, to install the latest `mmcv-full` with `CUDA 11` and `PyTorch 1.7.1`, use the following command:

    ```python
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html
    ```

  - Download this benchmark code.

    ```python
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection2.11-ShipRSImageNet
    ```

  - Install build requirements and then install MMDetection.

    ```python
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"
    ```

## Train with ShipRSImageNet

- Download  the ShipRSImageNet dataset. It is recommended to symlink the ShipRSImageNet  dataset root to  $mmdetection2.11-ShipRSImageNet/data:

  ```python
  ln -s $dataset/ShipRSImageNet/ $mmdetection2.11-ShipRSImageNet/data/
  ```

-  If your folder structure is different, you may need to change the corresponding paths in config files.

  - ```python
    mmdetection2.11-ShipRSImageNet
    ├── mmdet
    ├── tools
    ├── configs
    ├── data
    │   ├── ShipRSImageNet
    │   │   ├── COCO_Format
    │   │   ├── masks
    │   │   ├── VOC_Format
    │   │   │   ├── annotations
    │   │   │   ├── ImageSets
    │   │   │   ├── JPEGImages
    
    ```

- Prepare a config file:

  - The benchamark config file of ShipRSImageNet already in the following:

    - ```python
      $mmdetection2.11-ShipRSImageNet/configs/ShipRSImageNet/
      ```

      

- Example of train a model with ShipRSImageNet:

  - ```python
    python tools/train.py configs/ShipRSImageNet/faster_rcnn/faster_rcnn_r50_fpn_100e_ShipRSImageNet_Level0.py
    ```

  

# Models trained on ShipRSImageNet

We introduce two tasks: detection with horizontal bounding boxes (HBB for short) and segmentation with oriented bounding boxes (SBB for short). HBB aims at extracting bounding boxes with the same orientation of the image, it is an Object Detection task. SBB aims at semantically segmenting the image, it is a Semantic Segmentation task. 

The evaluation protocol follows the same mAP and mAR of area small/medium/large and mAP(@IoU=0.50:0.95) calculation used by MS-COCO. 


## Level 0

| Model                      | Backbone | Style   | HBB mAP | SBB mAP | Extraction code |Download |
| -------------------------- | -------- | ------- | -------- | ------ | -------- |------- |
| Faster RCNN with FPN       | R-50     | Pytorch | 0.550 |        | 2vrm |[model](https://pan.baidu.com/s/1bAvxP26OhdZM8gTSNGqZow)|
| Faster RCNN with FPN       | R-101    | Pytorch | 0.546 |        | f362 |[model](https://pan.baidu.com/s/1T0iqCfrLcOOPpv0k6YSL5Q)|
| Mask RCNN with FPN         | R-50     | Pytorch | 0.566 | 0.440 | 24eq |[model](https://pan.baidu.com/s/1sE_HngC0vlng61FUv_-HtQ)|
| Mask RCNN with FPN         | R-101    | Pytorch | 0.557 | 0.436 | lbcb |[model](https://pan.baidu.com/s/1tfaY8_8SUtWFbqghbNav8A)|
| Cascade Mask RCNN with FPN | R-50     | Pytorch | 0.568 | 0.430 | et6m |[model](https://pan.baidu.com/s/1wVOb8MS2ZItWJ-w3HZHF9A)|
| SSD                        | VGG16    | Pytorch | 0.464 |        | qabf |[model](https://pan.baidu.com/s/1Yj0F20PJr9e2op0rx8vdUw)|
| Retinanet with FPN         | R-50     | Pytorch | 0.418 |        | 7qdw |[model](https://pan.baidu.com/s/1nZC2UKqnS0hzdVP_sRXuBQ)|
| Retinanet with FPN         | R-101    | Pytorch | 0.419 |        | vdiq |[model](https://pan.baidu.com/s/1nMSEoDCAriiruEYnB-q4oA)|
| FoveaBox                   | R-101    | Pytorch | 0.453 |        | urbf            | [model](https://pan.baidu.com/s/13VPP1lmoAFaK-VR0S0nUZQ) |
| FCOS with FPN              | R-101    | Pytorch | 0.333 |        | 94ub |[model](https://pan.baidu.com/s/1qL-8i05OG80jqRTVQQW9HQ)|
|                            |          |         |          |        |          ||

## Level 1

| Model                      | Backbone | Style   | HBB mAP | SBB mAP | Extraction code |Download |
| -------------------------- | -------- | ------- | -------- | ------ | -------- |------- |
| Faster RCNN with FPN       | R-50     | Pytorch | 0.366 | - | 5i5a |[model](https://pan.baidu.com/s/1ofNMGBchAkg26IaO1TnjyA)|
| Faster RCNN with FPN       | R-101    | Pytorch | <u>0.461</u> | - | 6ts7 |[model](https://pan.baidu.com/s/1uBAoFgEjXBavvQg5c_uVcA)|
| Mask RCNN with FPN         | R-50     | Pytorch | <u>0.456</u> | 0.347 | 9gnt |[model](https://pan.baidu.com/s/1ViJGBtE6z4udATzsQU7AlQ)|
| Mask RCNN with FPN         | R-101    | Pytorch | 0.472 | 0.371 | wc62 |[model](https://pan.baidu.com/s/18LzR9Yek6TJivBns8_QGPA)|
| Cascade Mask RCNN with FPN | R-50     | Pytorch | 0.485 | 0.365 | a8bl |[model](https://pan.baidu.com/s/12rVdQCiApQFC9SG0NI0TLQ)|
| SSD                        | VGG16    | Pytorch | 0.397 | - | uffe |[model](https://pan.baidu.com/s/19H43Hbi1gI3n9Rq-BcZh6Q)|
| Retinanet with FPN         | R-50     | Pytorch | 0.368 | - | lfio |[model](https://pan.baidu.com/s/1SuhdUEoeACfTk8qUe48sbw)|
| Retinanet with FPN         | R-101    | Pytorch | 0.359 | - | p1rd |[model](https://pan.baidu.com/s/1Qeu4jWH1YaJaov7WbukS4w)|
| FoveaBox                   | R-101    | Pytorch | 0.389 | - | kwiq |[model](https://pan.baidu.com/s/12rKJ3HEVN_qGeFjabQabFg)|
| FCOS with FPN              | R-101    | Pytorch | 0.351 | - | 1djo |[model](https://pan.baidu.com/s/1bWn3N9THIk5_5vdGrMy6Sw)|
|                            |          |         |          |        |          ||

## Level 2

| Model                      | Backbone | Style   | HBB mAP | SBB mAP | Extraction code |Download |
| -------------------------- | -------- | ------- | -------- | ------ | -------- |------- |
| Faster RCNN with FPN       | R-50     | Pytorch | 0.345 | - | 924l |[model](https://pan.baidu.com/s/1AUzF2ZAPkLeNWBvQfDFpKw)|
| Faster RCNN with FPN       | R-101    | Pytorch | 0.479 | - | fb1b |[model](https://pan.baidu.com/s/1tDWOnOSGEUdIjI4HUwTZPQ)|
| Mask RCNN with FPN         | R-50     | Pytorch | 0.468 | 0.377 | so8j |[model](https://pan.baidu.com/s/1g35MRwqqsRmV7JOgoTWjqw)|
| Mask RCNN with FPN         | R-101    | Pytorch | 0.488 | 0.398 | 7q1g |[model](https://pan.baidu.com/s/1MGu88cRwzgmwJCg1WJz0mw)|
| Cascade Mask RCNN with FPN | R-50     | Pytorch | 0.492 | 0.389 | t9gr |[model](https://pan.baidu.com/s/1G4qqLwKWp4AlHXPUohSg2A)|
| SSD                        | VGG16    | Pytorch | 0.423 | - | t1ma |[model](https://pan.baidu.com/s/1N7Gt2EmFZue54DmZhW8y9g)|
| Retinanet with FPN         | R-50     | Pytorch | 0.369 | - | 4h0o |[model](https://pan.baidu.com/s/1rPLxArNCKn0P0oJGpq8qog)|
| Retinanet with FPN         | R-101    | Pytorch | 0.411 | - | g9ca |[model](https://pan.baidu.com/s/1UYnDcvyb_p9m2h7K_qL1iw)|
| FoveaBox                   | R-101    | Pytorch | 0.427 | - | 8e12 |[model](https://pan.baidu.com/s/1qztaomRQXp6l5nVrbbmB4g)|
| FCOS with FPN              | R-101    | Pytorch | 0.431 | - | 0hl0 |[model](https://pan.baidu.com/s/1IK3GYZb572PAOCJWierdAg)|
|                            |          |         |          |        |          ||
## Level 3

| Model                      | Backbone | Style   | HBB mAP | SBB mAP | Extraction code |Download |
| -------------------------- | -------- | ------- | -------- | :----: | -------- |------- |
| Faster RCNN with FPN       | R-50     | Pytorch | 0.375 | - | 7qmo |[model](https://pan.baidu.com/s/1ljwKD3_khLavvSiVSEOd5Q )|
| Faster RCNN with FPN       | R-101    | Pytorch | 0.543 | - | bmla |[model](https://pan.baidu.com/s/1SQHxti69NukyWopQS1NslQ)|
| Mask RCNN with FPN         | R-50     | Pytorch | 0.545 | 0.450 | a73h |[model](https://pan.baidu.com/s/1RbkByB2bo-_ubb5J67puyA)|
| Mask RCNN with FPN         | R-101    | Pytorch | 0.564 | 0.472 | 7k9i |[model](https://pan.baidu.com/s/1Hs7Fckr3l9jiZG22vSVZgg)|
| Cascade Mask RCNN with FPN | R-50     | Pytorch | 0.593 | 0.483 | ebga |[model](https://pan.baidu.com/s/1eJynOMggSJSqW1tIkktnxg)|
| SSD                        | VGG16    | Pytorch | 0.483 | - | otu5 |[model](https://pan.baidu.com/s/1FmEcAGaJJnXtBA63jW9k9w)|
| Retinanet with FPN         | R-50     | Pytorch | 0.326 | - | tu5a |[model](https://pan.baidu.com/s/11s8x7W35G7krMzQiJPCnPg)|
| Retinanet with FPN         | R-101    | Pytorch | 0.483 | - | ptv0 |[model](https://pan.baidu.com/s/1KWx7g3bcSAGOsOVMJr36TA)|
| FoveaBox                   | R-101    | Pytorch | 0.459 | - | 1acn |[model](https://pan.baidu.com/s/1p5ebAXwajj_A4s4HfqHfEw)|
| FCOS with FPN              | R-101    | Pytorch | 0.498 | - | 40a8 |[model](https://pan.baidu.com/s/11tNLbl2AgnHp-hLgy5yovg)|
|                            |          |         |          |        |          ||

# Development kit

The [ShipRSImageNet Development kit](https://github.com/zzndream/ShipRSImageNet_devkit) is based on DOTA [Development kit](https://github.com/CAPTAIN-WHU/DOTA_devkit)  and provides the following function

- Load and image, and show the bounding box on it.

- Covert VOC format label to COCO format label.    


# Citation

If you make use of the ShipRSImageNet  dataset, please cite our following paper:

```
Z. Zhang, L. Zhang, Y. Wang, P. Feng and R. He, "ShipRSImageNet: A Large-Scale Fine-Grained Dataset for Ship Detection in High-Resolution Optical Remote Sensing Images," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 8458-8472, 2021, doi: 10.1109/JSTARS.2021.3104230.
```
# Contact

If you have any the problem or feedback in using ShipRSImageNet, please contact:

- Zhengning Zhang at **23880666@qq.com**

# License

ShipRSImageNet is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.



_base_ = [
    './faster_rcnn_r50_fpn_100e_ShipRSImageNet_Level2.py'
]

model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
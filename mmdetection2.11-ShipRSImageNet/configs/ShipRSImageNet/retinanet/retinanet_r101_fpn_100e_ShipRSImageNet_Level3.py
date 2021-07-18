
_base_ = './retinanet_r50_fpn_100e_ShipRSImageNet_Level3.py'


model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

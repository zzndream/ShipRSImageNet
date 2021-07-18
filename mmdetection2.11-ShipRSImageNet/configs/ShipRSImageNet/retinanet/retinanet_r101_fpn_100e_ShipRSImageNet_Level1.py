
_base_ = './retinanet_r50_fpn_100e_ShipRSImageNet_Level1.py'


model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2)

optimizer = dict(type='SGD', lr=0.02/8, momentum=0.9, weight_decay=0.0001)
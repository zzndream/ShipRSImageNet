_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/ShipRSImageNet_Level0_instance.py',
    '../_base_/schedules/schedule_100e.py', '../_base_/default_runtime.py',
]


model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=2),
        mask_head=dict(
            num_classes=2)))


optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
evaluation = dict(interval=10, metric=['bbox', 'segm'])
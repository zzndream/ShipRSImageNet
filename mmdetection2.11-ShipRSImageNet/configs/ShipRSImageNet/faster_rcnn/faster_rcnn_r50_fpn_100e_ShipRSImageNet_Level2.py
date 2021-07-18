_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/ShipRSImageNet_Level2_detection.py',
    '../_base_/schedules/schedule_100e.py', '../_base_/default_runtime.py',
]


model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=25))
    )



evaluation = dict(interval=50, metric='bbox')
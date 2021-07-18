dataset_type = 'ShipRSImageNet_Level2'
# data_root = 'data/Ship_ImageNet/'
data_root = './data/ShipRSImageNet/'

CLASSES = ('Other Ship', 'Other Warship', 'Submarine', 'Aircraft Carrier', 'Cruiser', 'Destroyer',
           'Frigate', 'Patrol', 'Landing', 'Commander', 'Auxiliary Ships', 'Other Merchant',
           'Container Ship', 'RoRo', 'Cargo', 'Barge', 'Tugboat', 'Ferry', 'Yacht',
           'Sailboat', 'Fishing Vessel', 'Oil Tanker', 'Hovercraft', 'Motorboat', 'Dock',)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + 'COCO_Format/ShipRSImageNet_bbox_train_level_2.json',
        img_prefix=data_root + 'VOC_Format/JPEGImages/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + 'COCO_Format/ShipRSImageNet_bbox_val_level_2.json',
        img_prefix=data_root + 'VOC_Format/JPEGImages/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + 'COCO_Format/ShipRSImageNet_bbox_val_level_2.json',
        img_prefix=data_root + 'VOC_Format/JPEGImages/',
        pipeline=test_pipeline))

evaluation = dict(interval=10, metric=['bbox', 'segm'])
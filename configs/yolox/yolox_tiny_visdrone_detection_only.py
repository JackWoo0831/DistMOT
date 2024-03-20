_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
]

# define custom imports 
import sys, os
sys.path.append(os.getcwd())
custom_imports = dict(
    imports=['datasets.visdrone', 
             'datasets.visdrone_det'],
    allow_failed_imports=False)

# model settings
model = dict(
    type='YOLOX',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=32,
        ),
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=0.33,
        widen_factor=0.375,
        out_indices=(2, 3, 4),
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
    ),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=5,
        in_channels=96,
        feat_channels=96,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))


# dataset settings
dataset_type = 'VisDroneDetDataset'
data_root = '/data/wujiapeng/datasets/VisDrone2019/VisDrone2019'

img_scale = (1280, 704)  # width, height

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True), 

    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),

    dict(type='Resize', scale=img_scale, keep_ratio=True),
    
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile',),
    dict(type='Resize', scale=img_scale, keep_ratio=True),

    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, 
        data_root=data_root, 
        ann_file='/data/wujiapeng/datasets/VisDrone2019/VisDrone2019/annotations/VisDrone2019-MOT-train_qdtrack.json', 
        data_prefix=dict(img=''), 
        metainfo=dict(classes=('pedestrian', 'car', 'van', 'truck', 'bus')), 
        pipeline=train_pipeline
    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler'),  # image-based sampling
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/data/wujiapeng/datasets/VisDrone2019/VisDrone2019/annotations/VisDrone2019-MOT-val_qdtrack.json',
        data_prefix=dict(img=''), 
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='/data/wujiapeng/datasets/VisDrone2019/VisDrone2019/annotations/VisDrone2019-MOT-val_qdtrack.json',
    metric='bbox',
    )
test_evaluator = val_evaluator

# training settings
max_epochs = 50
num_last_epochs = 5
interval = 10

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

# optimizer
base_lr = 0.01
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# learning rate
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=3,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=3,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]

load_from = 'ckpts/yolox/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
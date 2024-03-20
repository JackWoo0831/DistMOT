_base_ = ['../yolox/yolox_l_visdrone_detection_only.py']

detector = _base_.model
detector.pop('data_preprocessor')
detector.test_cfg.nms.update(dict(iou_threshold=0.7))
del _base_.model

custom_imports = dict(
    imports=['datasets.visdrone', 
             'datasets.visdrone_det',
             'hooks.mot_save_result_hook'],
    allow_failed_imports=False)

model = dict(
    type='ByteTrack',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        pad_size_divisor=32,
        # in bytetrack, we provide joint train detector and evaluate tracking
        # performance, use_det_processor means use independent detector
        # data_preprocessor. of course, you can train detector independently
        # like strongsort
        use_det_processor=True,
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(576, 1024),
                size_divisor=32,
                interval=10)
        ]),
    detector=detector,
    tracker=dict(
        type='ByteTracker',
        motion=dict(type='KalmanFilter'),
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30))

# dataset settings
dataset_type = 'VisDroneDataset'
data_root = '/data/wujiapeng/datasets/VisDrone2019/VisDrone2019'

img_scale = (1600, 896)  # width, height

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
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=img_scale, keep_ratio=True),
            dict(type='LoadTrackAnnotations'),
        ]),
    dict(type='PackTrackInputs')
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
    sampler=dict(type='TrackImgSampler'),  # image-based sampling
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/data/wujiapeng/datasets/VisDrone2019/VisDrone2019/annotations/VisDrone2019-MOT-val_qdtrack.json',
        data_prefix=dict(img=''), 
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    dict(type='CocoVideoMetric', metric=['bbox'], classwise=True),
    # dict(type='MOTChallengeMetric', format_only=True, outfile_prefix='./visdrone_test_res')
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
        priority=49), 
    dict(
        type='TrackVisualizationHook', 
        draw=True, 
        frame_interval=1, 
        test_out_dir='/data/wujiapeng/codes/ctrMOT/qdtrack_baseline_vis'
    ), 
    dict(
        type='MotSaveResultHook', 
        save_dir='./txt_results', 
    )
]

load_from = 'ckpts/yolox/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
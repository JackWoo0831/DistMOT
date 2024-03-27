_base_ = [
    '../_base_/default_runtime.py'
]

# define custom imports 
import sys, os
sys.path.append(os.getcwd())
custom_imports = dict(
    imports=['datasets.visdrone', 
             'datasets.samplers',
             'models.distmot', 
             'hooks.mot_save_result_hook',
             'task_modules.assigners', 
             'task_modules.samplers'],
    allow_failed_imports=False)


yolox = dict(
    type='YOLOX',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        # mean=[103.530, 116.280, 123.675],  # YOLOX need norm
        # std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=1.0,
        widen_factor=1.0,
        out_indices=(2, 3, 4),
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
    ),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')),
    bbox_head=dict(
        type='YOLOXHead_woNMS',
        num_classes=5,
        in_channels=256,
        feat_channels=256,
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
    test_cfg=dict(score_thr=0.001, nms=dict(type='nms', iou_threshold=0.65)))


# model setting
detector = yolox
detector.pop('data_preprocessor')

model = dict(
    type='DistMOT',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        pad_size_divisor=32,
        ),
    detector=detector,
    track_head=dict(
        type='DistMOTHead',
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8, 16, 32]),
        embed_head=dict(
            type='DistEmbedHead',
            num_convs=4,
            num_fcs=1,
            in_channels=256,
            embed_channels=128,
            conv_out_channels=128, 
            norm_cfg=dict(type='GN', num_groups=32),
            loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.25),
            loss_track_aux=dict(
                type='MarginL2Loss',
                neg_pos_ub=3,
                pos_margin=0,
                neg_margin=0.1,
                hard_mining=True,
                loss_weight=1.0),
            mask_cfg=dict(
                type='MaskGenerator2D_MixAttn',
                feat_dim=128, 
            ), 
            memo_cfg=dict(max_size=5, seq_num=56)
            ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner_track',
                pos_iou_thr=0.7,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='CombinedSampler_track',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=3,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(type='RandomSampler')))),
    tracker=dict(
        type='DistMOTTracker',
        init_score_thr=0.1,
        obj_score_thr=0.1,
        match_score_thr=0.5,
        memo_tracklet_frames=30,
        memo_backdrop_frames=1,
        memo_momentum=1.0,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=False,
        match_metric='dotproduct'), 
    )
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))
# learning policy
param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True, milestones=[10])
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=4)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    visualization=dict(type='TrackVisualizationHook', draw=False))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TrackLocalVisualizer', vis_backends=vis_backends, name='visualizer')


# dataset settings
dataset_type = 'VisDroneDataset'
data_root = '/data/wujiapeng/datasets/VisDrone2019/VisDrone2019'
img_scale = (1600, 896) 

backend_args = None
# data pipeline
train_pipeline = [
    dict(
        type='UniformRefFrameSample',
        num_ref_imgs=1,
        frame_range=10,
        filter_key_img=True),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadTrackAnnotations'),
            dict(
                type='RandomResize',
                scale=img_scale,
                ratio_range=(0.8, 1.2),
                keep_ratio=True,
                clip_object_border=False),
            dict(type='PhotoMetricDistortion')
        ]),
    dict(
        type='TransformBroadcaster',
        # different cropped positions for different frames
        share_random_params=False,
        transforms=[
            dict(
                type='RandomCrop', crop_size=img_scale, bbox_clip_border=False)
        ]),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='RandomFlip', prob=0.5),
        ]),
    dict(type='PackTrackInputs')
]

test_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='Resize', scale=img_scale, keep_ratio=True),
            dict(type='LoadTrackAnnotations')
        ]),
    dict(type='PackTrackInputs')
]

# dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='TrackImgSampler'),  # image-based sampling
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        visibility_thr=-1,
        ann_file='/data/wujiapeng/datasets/VisDrone2019/VisDrone2019/annotations/VisDrone2019-MOT-train_qdtrack.json',
        data_prefix=dict(img_path=''),
        metainfo=dict(classes=('pedestrian', 'car', 'van', 'truck', 'bus')), 
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='TrackImgSampler'),  # image-based sampling
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/data/wujiapeng/datasets/VisDrone2019/VisDrone2019/annotations/VisDrone2019-MOT-val_qdtrack.json',
        data_prefix=dict(img_path=''),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='TrackImgSampler'),  # image-based sampling
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/data/wujiapeng/datasets/VisDrone2019/VisDrone2019/annotations/VisDrone2019-MOT-val_qdtrack.json',
        data_prefix=dict(img_path=''),
        test_mode=True,
        pipeline=test_pipeline))

# evaluator
val_evaluator = dict(
    dict(type='CocoVideoMetric', metric=['bbox'], classwise=True),
    # dict(type='MOTChallengeMetric', format_only=True, outfile_prefix='./visdrone_test_res')
    )
test_evaluator = val_evaluator

custom_hooks = [
    dict(type='SyncBuffersHook'), 
    # dict(
    #     type='TrackVisualizationHook', 
    #     draw=True, 
    #     frame_interval=1, 
    #     score_thr=0.1, 
    #     test_out_dir='/data/wujiapeng/codes/ctrMOT/qdtrack_baseline_vis'
    # ), 
    dict(
        type='MotSaveResultHook', 
        save_dir='./txt_results', 
    ), 
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]

load_from = 'ckpts/yolox/yolox_large_VisDrone_20epochs_20240225.pth'
_base_ = ['../bytetrack/bytetrack_yolox_tiny_uavdt.py']

model = dict(
    type='OCSORT',
    tracker=dict(
        _delete_=True,
        type='OCSORTTracker',
        motion=dict(type='KalmanFilter'),
        obj_score_thr=0.1,
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thr=0.3,
        num_tentatives=3,
        vel_consist_weight=0.2,
        vel_delta_t=3,
        num_frames_retain=30))


custom_hooks = [
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49), 
    # dict(
    #     type='TrackVisualizationHook', 
    #     draw=True, 
    #     frame_interval=1, 
    #     test_out_dir='/data/wujiapeng/codes/ctrMOT/qdtrack_baseline_vis'
    # ), 
    dict(
        type='MotSaveResultHook', 
        save_dir='./txt_results_ocsort', 
        dataset_type='mot', 
        video_name_pos_in_path=-3
    )
]
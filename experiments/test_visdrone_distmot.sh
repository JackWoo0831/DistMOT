CUDA_VISIBLE_DEVICES=0 python tools/test_tracking.py configs/distmot/distmot_yolox_tiny_visdrone.py --checkpoint work_dirs/tttest_4/epoch_8.pth
# CUDA_VISIBLE_DEVICES=2 python tools/test_tracking.py configs/distmot/distmot_yolox_tiny_visdrone.py --checkpoint ckpts/distmot/distmot_yolox_tiny_visdrone_hist_memo_2dMask/epoch_4.pth
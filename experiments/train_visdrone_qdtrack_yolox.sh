# CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/qdtrack/qdtrack_yolox_visdrone_baseline.py --resume work_dirs/qdtrack_yolox_visdrone_baseline/epoch_3.pth
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/qdtrack/qdtrack_yolox_tiny_visdrone_baseline.py

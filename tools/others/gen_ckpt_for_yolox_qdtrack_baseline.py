"""
Convert origin yolox ckpt to fit qdtrack
Mainly add 'detector'
"""

import torch

def main():
    yolox_ckpt = 'work_dirs/yolox_tiny_uavdt_detection_only/epoch_20.pth'
    target_ckpt = 'ckpts/yolox/yolox_tiny_UAVDT_20epochs_20240316.pth'

    new_ckpt = dict()
    
    yolox_ckpt = torch.load(yolox_ckpt)
    new_ckpt['state_dict'] = dict()

    new_ckpt['meta'] = yolox_ckpt['meta']

    for k, v in yolox_ckpt['state_dict'].items():

        new_k = 'detector.' + k
        new_ckpt['state_dict'][new_k] = yolox_ckpt['state_dict'][k]


    torch.save(new_ckpt, target_ckpt)

    print(new_ckpt['state_dict'].keys())


if __name__ == '__main__':
    main()

    # python tools/others/gen_ckpt_for_yolox_qdtrack_baseline.py
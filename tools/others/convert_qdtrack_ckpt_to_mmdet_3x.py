"""
Convert the checkpoint of qdtrack from origin version (mmdet 2.x) to newer mmdet 3.x
mainly replace the key 'track_head.track_head' to 'track_head.embed_head'
"""

import torch 
import copy

def main():
    ckpt = 'ckpts/qdtrack/qdtrack-frcnn_r50_fpn_12e_mot_bdd100k.pth'
    ckpt = torch.load(ckpt)

    new_ckpt = dict()
    new_ckpt['meta'] = ckpt['meta']
    new_ckpt['state_dict'] = dict()

    for k, v in ckpt['state_dict'].items():
        new_k = k.replace('track_head.track_head', 'track_head.embed_head')

        new_ckpt['state_dict'][new_k] = ckpt['state_dict'][k]

    torch.save(new_ckpt, 'ckpts/qdtrack/qdtrack-frcnn_r50_fpn_12e_mot_bdd100k_mmdet3.pth')

    print(new_ckpt['state_dict'].keys())


if __name__ == '__main__':
    main()

    # python tools/others/convert_qdtrack_ckpt_to_mmdet_3x.py

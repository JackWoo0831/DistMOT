"""
save img with bboxes from result txt
only support visdrone yet
"""

import os 
import os.path as osp 
import cv2 
import numpy as np 
import torch 

from mmengine.structures.instance_data import InstanceData
from mmdet.visualization.local_visualizer import TrackLocalVisualizer
import mmcv 

DATA_ROOT = '/data/wujiapeng/datasets/VisDrone2019/VisDrone2019/VisDrone2019-MOT-val/sequences'
txt_results_folder = './txt_results'
OUTPUT_FOLDER = './qdtrack_baseline_vis'

CERTAIN_SEQS = []

visualizer = TrackLocalVisualizer()
visualizer.dataset_meta = dict(classes=('pedestrian', 'car', 'van', 'truck', 'bus'))

def draw_a_seq(seq_name):
    imgs_path = osp.join(DATA_ROOT, seq_name, )
    imgs = sorted(os.listdir(imgs_path))

    num_of_frames = len(imgs)
    seq_result = np.loadtxt(fname=osp.join(txt_results_folder, f'{seq_name}.txt'), delimiter=',', dtype=float)

    for frame_idx in range(1, num_of_frames + 1):
        img_path = osp.join(DATA_ROOT, seq_name, imgs[frame_idx - 1])
        raw_image = mmcv.imread(img_path, channel_order='rgb')
        
        frame_result = seq_result[seq_result[:, 0] == frame_idx]

        if frame_result.shape[0] > 0:
            ids = torch.from_numpy(frame_result[:, 1]).int()
            bboxes = torch.from_numpy(frame_result[:, 2: 6])
            classes = torch.from_numpy(frame_result[:, -3]).int()
            scores = torch.from_numpy(frame_result[:, -4])

            # convert bboxes format
            bboxes[:, 2] += bboxes[:, 0]
            bboxes[:, 3] += bboxes[:, 1]

            # align classes 
            classes[classes == 1] = 0
            classes[classes == 4] = 1
            classes[classes == 5] = 2
            classes[classes == 6] = 3
            classes[classes == 9] = 4

            cur_ins_data = InstanceData(instances_id=ids, 
                                        bboxes=bboxes, 
                                        labels=classes, 
                                        scores=scores)
            
            drawn_img = visualizer._draw_instances(
                image=raw_image, 
                instances=cur_ins_data
            )

            out_file = osp.join(OUTPUT_FOLDER, seq_name, imgs[frame_idx - 1])
            mmcv.imwrite(drawn_img[..., ::-1], out_file)


def draw():
    
    seq_names = []
    if not len(CERTAIN_SEQS):
        results = os.listdir(txt_results_folder)
        for res in results:
            seq_name = res[:-4]
            seq_names.append(seq_name)

    else:
        seq_names = CERTAIN_SEQS

    print('Following seqs will be drawn: ', seq_names)

    for seq_name in seq_names:
        print(f'processing {seq_name}')
        draw_a_seq(seq_name)


if __name__ == '__main__':
    draw()

    # python tools/analysis_tools/mot/vis_result_from_txt.py
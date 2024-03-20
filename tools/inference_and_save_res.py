"""
This code supports inference several seqs and save the results to txt in MOT format.
without using MOTChallengeMetric and installing TrackEval.
"""

import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import mmcv
import mmengine
from mmengine.registry import init_default_scope

from mmdet.apis import inference_mot, init_track_model

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')
SAVE_FORMAT = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, help='Input folders that contain the seqs. for example: MOT17/test/')
    parser.add_argument('config', help='config file')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--detector', help='det checkpoint file')
    parser.add_argument('--reid', help='reid checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='The threshold of score to filter bboxes.')
    parser.add_argument(
        '--out', help='output txt file. the result will be: $out/$seq_name.txt')
    args = parser.parse_args()
    return args


def main(args):

    assert osp.isdir(args.inputs)

    seqs = sorted(os.listdir(args.inputs))  # all sequences

    for seq in seqs:  # handle every seq
        print(f'\nprocessing {seq}\n')

        # init model every time
        init_default_scope('mmdet')

        model = init_track_model(
        args.config,
        args.checkpoint,
        args.detector,
        args.reid,
        device=args.device)

        imgs = sorted(
            filter(lambda x: x.endswith(IMG_EXTENSIONS),
                   os.listdir(osp.join(args.inputs, seq))),
            key=lambda x: int(x.split('.')[0]))
        
        prog_bar = mmengine.ProgressBar(len(imgs))

        to_txt = osp.join(args.out, f'{seq}.txt')
        if not osp.exists(osp.join(args.out, )):
            os.makedirs(osp.join(args.out, ))
            

        with open(to_txt, 'w') as f:
            for i, img in enumerate(imgs):
    
                img_path = osp.join(args.inputs, seq, img)
                img = mmcv.imread(img_path)
                # result [TrackDataSample]
                result = inference_mot(model, img, frame_id=i, video_len=len(imgs))

                # get bboxes, cls_labels, ids

                result = result[0].pred_track_instances

                bboxes, labels, ids = result.bboxes, result.labels, result.instances_id
                scores = result.scores

                num_of_objs = bboxes.shape[0]

                for obj_idx in range(num_of_objs):
                    bbox, label, id = bboxes[obj_idx], labels[obj_idx], ids[obj_idx]
                    x1, y1 = bbox[0].item(), bbox[1].item()
                    x2, y2 = bbox[2].item(), bbox[3].item()

                    w, h = x2 - x1, y2 - y1 

                    f.write(SAVE_FORMAT.format(frame=i + 1, id=id, x1=x1, y1=y1, w=w, h=h))

                prog_bar.update()


if __name__ == '__main__':
    args = parse_args()
    main(args)

    # CUDA_VISIBLE_DEVICES=1 python tools/inference_and_save_res.py /data/wujiapeng/datasets/VisDrone2019/VisDrone2019/VisDrone2019-MOT-test-dev/sequences configs/qdtrack/qdtrack_visdrone_baseline.py --checkpoint ckpts/qdtrack/epoch_10.pth --out result_txt
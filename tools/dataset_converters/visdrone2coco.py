"""
covert VisDrone2019-MOT to COCO for mmdetection

"""

import argparse
import os
import os.path as osp
from collections import defaultdict

import mmengine
import numpy as np
from tqdm import tqdm
import cv2 

CATEGORIES = [1, 4, 5, 6, 9]
CAT_MAP = {item: idx + 1 for idx, item in enumerate(CATEGORIES)}
CERTAIN_SEQS = []

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert VisDrone to COCO-VID format.')
    parser.add_argument('-i', '--input', help='path of MOT data')
    parser.add_argument(
        '-o', '--output', help='path to save coco formatted label file')
    parser.add_argument(
        '--half', action='store_true', help='half train set'
    )
    return parser.parse_args()


def handle_a_seq(gts):
    """
    handle a gt of a seq
    Args:
        gts: List[str]
    Returns:
        Dict[frame_id: dict]
    """

    outputs = defaultdict(list)
    for gt in gts:
        gt = gt.strip().split(',')
        frame_id, ins_id = map(int, gt[:2])
        bbox = list(map(float, gt[2:6]))

        conf = float(gt[6])
        class_id = int(gt[7])
        visibility = 1. - float(gt[9]) / 3
        
        if not class_id in CATEGORIES: continue
        anns = dict(
            category_id=CAT_MAP[class_id],  # TODO
            bbox=bbox,
            area=bbox[2] * bbox[3],
            iscrowd=False,
            visibility=visibility,
            mot_instance_id=ins_id,
            mot_conf=conf,
            mot_class_id=class_id)
        outputs[frame_id].append(anns)
    return outputs


def main():
    """
    main func
    """
    args = parse_args()
    if not osp.exists(args.output):
        os.makedirs(args.output)

    sets = ['VisDrone2019-MOT-train', 'VisDrone2019-MOT-val', 'VisDrone2019-MOT-test-dev']

    vid_id, img_id, ann_id = 1, 1, 1

    for subset in sets:
        ins_id = 0
        print(f'Converting {subset} set to COCO format')

        in_folder = osp.join(args.input, subset)
        out_file = osp.join(args.output, f'{subset}_qdtrack.json')

        outputs = defaultdict(list)
        outputs['categories'] = [dict(id=1, name='pedestrian'), 
                                 dict(id=2, name='car'), 
                                 dict(id=3, name='van'), 
                                 dict(id=4, name='truck'), 
                                 dict(id=5, name='bus')]
        

        video_names = os.listdir(osp.join(in_folder, 'sequences'))
        video_names = sorted(video_names)

        if len(CERTAIN_SEQS):
            video_names = [name for name in video_names if name in CERTAIN_SEQS]
            print(f'Warning: Only {video_names} are convertered')

            if not len(video_names): continue

        for video_name in video_names:
            ins_maps = dict()  # gt track id -> anno track id

            img_names = os.listdir(osp.join(in_folder, 'sequences', video_name))
            img_names = sorted(img_names)

            # get width and height
            img_example = cv2.imread(osp.join(in_folder, 'sequences', video_name, img_names[0]))
            height, width = img_example.shape[0], img_example.shape[1]

            # add video anno
            video = dict(
                id=vid_id,
                name=video_name,
                fps=30,
                width=width,
                height=height)
            

            gts = mmengine.list_from_file(osp.join(in_folder, 'annotations', f'{video_name}.txt'))

            img2gts = handle_a_seq(gts)  # storage anno info
            valid_frame_ids = img2gts.keys()

            if 'train' in subset and args.half:
                img_names = img_names[:len(img_names) // 2]

            # add image and instance anno
            
            for frame_id, name in enumerate(img_names):

                # filter the frames that contain no annotations
                if not (frame_id + 1) in valid_frame_ids:
                    print(f'Warning! Video {video_name} frame {frame_id + 1} contains no data')
                    continue

                img_name = osp.join(in_folder, 'sequences', video_name, name)
                mot_frame_id = int(name.split('.')[0])

                # add image anno
                image = dict(
                    id=img_id,
                    video_id=vid_id,
                    file_name=img_name,
                    height=height,
                    width=width,
                    frame_id=frame_id,
                    mot_frame_id=mot_frame_id)
                

                gts = img2gts[mot_frame_id]  # instances that in current frame

                for gt in gts:
                    gt.update(id=ann_id, image_id=img_id)
                    mot_ins_id = gt['mot_instance_id']

                    if mot_ins_id in ins_maps:
                        gt['instance_id'] = ins_maps[mot_ins_id]
                    else:
                        gt['instance_id'] = ins_id
                        ins_maps[mot_ins_id] = ins_id
                        ins_id += 1
                    outputs['annotations'].append(gt)
                    ann_id += 1

                outputs['images'].append(image)
                img_id += 1

            outputs['videos'].append(video)
            vid_id += 1
            outputs['num_instances'] = ins_id
            
        print(f'{subset} has {ins_id} instances.')
        mmengine.dump(outputs, out_file)
        
        print('Done!')         



if __name__ == '__main__':
    main()
    
    # python tools/dataset_converters/visdrone2coco.py --input /data/wujiapeng/datasets/VisDrone2019/VisDrone2019 --output /data/wujiapeng/datasets/VisDrone2019/VisDrone2019/annotations --half

"""
Hook responsible for saving txt result 
Call 'after_test_iter'
"""
from typing import Sequence
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from mmdet.structures.track_data_sample import TrackDataSample

import os 

@HOOKS.register_module()
class MotSaveResultHook(Hook):
    def __init__(self, 
                 save_dir: str, 
                 dataset_type: str = 'visdrone',  # write format, mot, visdrone or kitti,
                 video_name_pos_in_path: int = -2,  # video name in the position of img path splitted by '/'
                 ) -> None:
        super().__init__()

        self.save_dir = save_dir
        self.video_name_pos_in_path = video_name_pos_in_path

        if dataset_type == 'mot':
            self.save_format ='{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
            self.CLS_DICT = {
                0: 0
            }
        elif dataset_type == 'kitti':
            self.save_format = '{frame} {id} {label} -1 -1 -1 {x1} {y1} {x2} {y2} -1 -1 -1 -1000 -1000 -1000 -10 1\n'
            self.CLS_DICT = {
                1: 'Pedestrian', 
                2: 'Car'
            }
        elif dataset_type == 'visdrone':
            self.save_format = '{frame},{id},{x1},{y1},{w},{h},1,{label},0,0\n'
            self.CLS_DICT = {
                0: 1, 
                1: 4,
                2: 5,
                3: 6,
                4: 9,
            }
        else:
            raise NotImplementedError
        
        self.dataset_type = dataset_type
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def after_test_iter(self, runner, batch_idx: int, data_batch: dict, outputs: Sequence[TrackDataSample]) -> None:
        
        video_result = outputs[0]  # TrackDataSample, result of current video
        track_data_sample = data_batch['data_samples'][0]
        img_eg = track_data_sample[0]

        video_name = img_eg.img_path.split('/')[self.video_name_pos_in_path]
        to_txt = os.path.join(self.save_dir, video_name + '.txt')

        video_len = len(video_result)

        with open(to_txt, 'a') as f:
            
            for frame_id in range(video_len):  # in inference stage, video_len == 1
                frame_result = video_result[frame_id]

                actual_frame_id = frame_result.frame_id + 1  # frame id in total sequence
                frame_result = frame_result.pred_track_instances  # InstanceData
                # keys of frame_result: ['bboxes', 'labels', 'instances_id', 'scores']

                bboxes = frame_result.bboxes.cpu()  # tlbr
                labels = frame_result.labels.cpu() 
                instance_id = frame_result.instances_id.cpu()
                scores = frame_result.scores.cpu()

                num_of_objects = bboxes.shape[0]

                for obj_id in range(num_of_objects):
                    
                    x0, y0, x1, y1 = bboxes[obj_id, 0], bboxes[obj_id, 1], bboxes[obj_id, 2], bboxes[obj_id, 3]
                    
                    w, h = x1 - x0, y1 - y0

                    label = self.CLS_DICT.get(labels[obj_id].item(), 0)

                    line = self.save_format.format(frame=actual_frame_id, id=instance_id[obj_id], x1=x0, y1=y0, w=w, h=h,
                                                   x2=x1, y2=y1, label=label)
                    
                    f.write(line)
        f.close()

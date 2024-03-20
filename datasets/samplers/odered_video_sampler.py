"""
Make sure the data fed into models obey following orders:
(video1, frame_n), ..., (video1, frame_x), ..., (video2, frame_), ..., (video_n, ...)
that is, the video id should in order but the frame id shoule be random
"""
import math
import random
from typing import Iterator, Optional, Sized

import numpy as np
from mmengine.dataset import ClassBalancedDataset, ConcatDataset
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmdet.registry import DATA_SAMPLERS

@DATA_SAMPLERS.register_module()
class OrderedVideoSampler(Sampler):
    def __init__(
        self,
        dataset: Sized,
        seed: Optional[int] = None,
    ) -> None:
        
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        if seed is None:
            self.seed = sync_random_seed()
        else:
            self.seed = seed

        self.dataset = dataset
        self.indices = []  # for test
        self.indices_train = dict()  # for train, key: video id, value: (video_id, frame_id)

        self.test_mode = self.dataset.test_mode
        num_videos = len(self.dataset)
        
        if self.test_mode:
            # in test mode, the images belong to the same video must be put
            # on the same device.
            if num_videos < self.world_size:
                raise ValueError(f'only {num_videos} videos loaded,'
                                    f'but {self.world_size} gpus were given.')
            chunks = np.array_split(
                list(range(num_videos)), self.world_size)
            for videos_inds in chunks:
                indices_chunk = []
                for video_ind in videos_inds:
                    indices_chunk.extend([
                        (video_ind, frame_ind) for frame_ind in range(
                            self.dataset.get_len_per_video(video_ind))
                    ])
                self.indices.append(indices_chunk)
        else:
            assert self.world_size == 1, 'distributed mode is not supported yet'
            for video_ind in range(num_videos):

                num_frames = self.dataset.get_len_per_video(video_ind)
                self.indices_train[video_ind] = [(video_ind, frame_ind) for frame_ind in range(num_frames)]

        if self.test_mode:
            self.num_samples = len(self.indices[self.rank])
            self.total_size = sum(
                [len(index_list) for index_list in self.indices])
        else:
            self.num_samples = sum([len(frames) for frames in self.indices_train.values()])
            self.total_size = self.num_samples * self.world_size

    def __iter__(self) -> Iterator:
        if self.test_mode:
            # in test mode, the order of frames can not be shuffled.
            indices = self.indices[self.rank]
        else:
            # deterministically shuffle based on epoch
            rng = random.Random(self.epoch + self.seed)
            indices = self._random_each_video(rng)

            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size

            # subsample
            indices = indices[self.rank:self.total_size:self.world_size]
            assert len(indices) == self.num_samples

        return iter(indices)
    
    def _random_each_video(self, rng: random.Random):

        ret = []
        
        for video_id in self.indices_train.keys():
            ret.extend(rng.sample(
                self.indices_train[video_id], 
                len(self.indices_train[video_id])
            ))

        return ret

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
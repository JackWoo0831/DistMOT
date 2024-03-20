"""
register the VisDrone dataset for detection
"""

import os.path as osp
from typing import List, Union

from mmdet.registry import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module()
class UAVDTDetDataset(CocoDataset):
    """Dataset for VisDrone-MOT.

    Args:
        visibility_thr (float, optional): The minimum visibility
            for the objects during training. Default to -1.
    """

    METAINFO = {
        'classes':
        ('car', ), 
        'palette':
        [(220, 20, 60), ]
    }
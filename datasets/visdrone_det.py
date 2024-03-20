"""
register the VisDrone dataset for detection
"""

import os.path as osp
from typing import List, Union

from mmdet.registry import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module()
class VisDroneDetDataset(CocoDataset):
    """Dataset for VisDrone-MOT.

    Args:
        visibility_thr (float, optional): The minimum visibility
            for the objects during training. Default to -1.
    """

    METAINFO = {
        'classes':
        ('pedestrian', 'car', 'van', 'truck', 'bus'), 
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),]
    }
"""
reid head following https://openaccess.thecvf.com/content/ICCV2023/supplemental/Liu_Uncertainty-aware_Unsupervised_Multi-Object_ICCV_2023_supplemental.pdf
"""

from typing import List, Tuple, Union, Optional

import math 
import torch 
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.structures import TrackSampleList
from mmdet.utils import OptConfigType, OptMultiConfig, ConfigType
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, gain=2, ) -> None:
        super().__init__()

        self.gain = gain

        self.proj = ConvModule(in_channels // 4, out_channels, kernel_size=3, 
                               stride=1, padding=1, norm_cfg=dict(type='BN', requires_grad=True),
                               act_cfg=dict(type='SiLU', inplace=True))
        
    def forward(self, x):
        # rearrange pixels
        x = F.pixel_shuffle(x, self.gain)  # (c, h, w) -> (c // gain^2, h * gain, w * gain)
        return self.proj(x)

@MODELS.register_module()
class REID_HEAD(BaseModule):
    def __init__(self,
                 input_img_size = (1600, 896), 
                 in_channels = (256, 512, 1024), 
                 strides = (8, 16, 32), 
                 widen_factor = 1.0, 
                 embedding_dim = 128, 
                 deformable = False,  # deformable to address occlusion， 
                 loss_reid = None,  # loss, OptConfigType
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.strides = strides
        self.widen_factor = widen_factor
        self.embedding_dim = embedding_dim

        self.loss = MODELS.build(loss_reid)

        self._init_layers()


    def _init_layers(self, ):

        self.reid_layers = []
        
        # layer for scale 0: (256 x h/8 x w/8)
        self.reid_layers.append(
            ConvModule(in_channels=int(self.in_channels[0] * self.widen_factor), 
                       out_channels=int(256 * self.widen_factor),
                       kernel_size=3, 
                       stride=1, 
                       padding=1, 
                       norm_cfg=dict(type='BN', requires_grad=True), 
                       act_cfg=dict(type='SiLU', inplace=True)),                        
        )

        # layer for scale 1: (512 x h/16 x w/16)
        self.reid_layers.append(
            Upsample(in_channels=int(self.in_channels[1] * self.widen_factor),
                     out_channels=int(256 * self.widen_factor), 
                     gain=2), 
        )

        # layer for scale 2: (1024 x h/32 x w/32)
        self.reid_layers.append(
            nn.Sequential(
                *[
                    Upsample(in_channels=int(self.in_channels[2] * self.widen_factor),
                             out_channels=int(self.in_channels[2] * self.widen_factor / 2), 
                             gain=2), 
                    Upsample(in_channels=int(self.in_channels[2] * self.widen_factor / 2),
                             out_channels=int(256 * self.widen_factor), 
                             gain=2), 
                ]
            )
        )

        # final layer, fuse multi-scale feats
        self.reid_layers.append(
            nn.Sequential(
                *[
                    ConvModule(in_channels=int(256 * self.widen_factor) * 3, 
                               out_channels=int(256 * self.widen_factor), 
                               kernel_size=3, 
                               padding=1, 
                               norm_cfg=dict(type='BN', requires_grad=True), 
                               act_cfg=dict(type='SiLU', inplace=True)), 
                    ConvModule(in_channels=int(256 * self.widen_factor), 
                               out_channels=self.embedding_dim, 
                               kernel_size=1, 
                               padding=0, 
                               act_cfg=None, )
                ]
            )
        )

        assert len(self.reid_layers) == 4

    def loss(self, multi_scale_feats: List[torch.Tensor], data_samples: TrackSampleList = None) -> dict:
        """
        Args:
            multi_scale_feats: mutli scale features output by YOLOX neck
            data_samples: contains gt info
        
        """
        mid_feats = []
        for idx, layer in enumerate(self.reid_layers):
            if idx < 3:
                mid_feats.append(layer(multi_scale_feats[idx]))
            else:
                # concat and final
                mid_feats = torch.cat(mid_feats, dim=1)
                mid_feats = layer(mid_feats)  # (bs, embedding_dim, h/8, w/8)

        reid_feats = F.normalize(mid_feats, dim=1)

        return dict()
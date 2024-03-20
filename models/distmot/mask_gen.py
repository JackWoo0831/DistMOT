"""
Generating masks for discriminative learning
Core codes
"""
from typing import List, Optional, Tuple

from mmengine.model import BaseModule
from torch import Tensor
import torch 
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

import matplotlib.pyplot as plt 

from mmdet.registry import MODELS

@MODELS.register_module()
class MaskGenerator1D(BaseModule):
    """1D Mask generator model"""
    def __init__(self, 
                 manner: str = 'self_attn',  # self_attn, conv
                 feat_dim: int = 256,  # feature dim
                 layer_num: int = 1,
                 norm: str = 'LayerNorm',  # norm layer
                 init_cfg: dict = dict(
                     type='Xavier',
                     layer='Linear',
                     distribution='uniform',
                     bias=0,
                     )):
        
        super().__init__(init_cfg)

        self.manner = manner
        if not self.manner in ['self_attn']: raise NotImplementedError
        self.feat_dim = feat_dim
        self.layer_num = layer_num 

        self.query = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        if norm == 'LayerNorm':
            norm_type = nn.LayerNorm
            params = dict(normalized_shape=self.feat_dim)
        elif norm == 'BatchNorm':
            norm_type = nn.BatchNorm1d
            params = dict(num_features=self.feat_dim)
        self.norm = nn.ModuleList()

        for layer_idx in range(layer_num):
            self.query.append(
                nn.Linear(in_features=self.feat_dim, 
                          out_features=self.feat_dim)
            )
            self.key.append(
                nn.Linear(in_features=self.feat_dim, 
                          out_features=self.feat_dim)
            )
            self.value.append(
                nn.Linear(in_features=self.feat_dim, 
                          out_features=self.feat_dim)
            )

            self.norm.append(norm_type(**params))

        self.final_fc = nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim)

    
    def forward(self, x: Tensor) -> Tensor:
        # self-attn * layers + sigmoid + residual structure
        # x: torch.Tensor, (num_of_rois, feat_dim)
        
        for layer_idx in range(self.layer_num):
            q = self.query[layer_idx](x)
            k = self.key[layer_idx](x)
            v = self.value[layer_idx](x)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.feat_dim ** 0.5)  # (num_of_rois, num_of_rois)
            attn_scores = F.softmax(attn_scores, dim=-1)

            x = torch.matmul(attn_scores, v) + x  # (num_of_rois, feat_dim)

            x = self.norm[layer_idx](x)

        ret = self.final_fc(x)

        return torch.sigmoid(ret)  # as a mask


@MODELS.register_module()
class MaskGenerator2D(BaseModule):
    """2D Mask generator model"""
    def __init__(self, 
                 feat_dim: int = 64,  # feature dim
                 size: int = 7,  # width and height of ROI
                 layer_num: int = None,  # ommited
                 conv_layer_num: int = 2,
                 attn_layer_num: int = 2, 
                 norm: str = 'LayerNorm',  # norm layer
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)

        self.feat_dim = feat_dim
        self.input_size = size 

        self.conv_layer_num = conv_layer_num
        self.convs = nn.ModuleList()
        for layer_idx in range(conv_layer_num):
            # (num_roi, c, h, w) -> (num_roi, c // 2, h // 2, w // 2)
            self.convs.append(
                ConvModule(
                    in_channels=self.feat_dim, 
                    out_channels=self.feat_dim // 2 if layer_idx == conv_layer_num - 1 else self.feat_dim, 
                    kernel_size=3,
                    stride=2,
                    padding=1, 
                    norm_cfg=dict(type='GN', num_groups=16)
                )
            )

            self.input_size = (self.input_size + 1) // 2

        # update feat_dim and size
        self.feat_dim = self.feat_dim // 2

        self.attn_layer_num = attn_layer_num
        self.query = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()
        self.norm = nn.ModuleList()

        if norm == 'GroupNorm':
            norm_type = nn.GroupNorm
            params = dict(num_groups=32, num_channels=self.feat_dim)
        elif norm == 'LayerNorm':
            norm_type = nn.LayerNorm
            params = dict(normalized_shape=[self.input_size * self.input_size, self.feat_dim])
        else: raise NotImplementedError

        for layer_idx in range(attn_layer_num):
            self.query.append(
                nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim)
            )
            self.key.append(
                nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim)
            )
            self.value.append(
                nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim)
            )

            self.norm.append(norm_type(**params))


        self.final_fc = nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim)
        
    def forward(self, x: Tensor):
        # x: (num_of_rois, dim, h, w)

        for layer_idx in range(self.conv_layer_num):
            x = self.convs[layer_idx](x)

        # x: (num_of_rois, dim, h, w)
            
        x = x.flatten(2)  # (num_of_rois, dim, hw)
        x = x.permute(0, 2, 1)  # (num_of_rois, hw, dim)

        for layer_idx in range(self.attn_layer_num):
            # conv
            q = self.query[layer_idx](x)  # (num_of_rois, hw, dim)
            k = self.key[layer_idx](x)
            v = self.value[layer_idx](x)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.feat_dim ** 0.5)  # (num_of_rois, hw, hw)
            attn_scores = F.softmax(attn_scores, dim=-1)  # (num_of_rois, hw, hw)

            x = x + torch.matmul(attn_scores, v)  # (num_of_rois, hw, dim)

            x = self.norm[layer_idx](x)  # (num_of_rois, hw, dim)

        ret = self.final_fc(x)  # (num_of_rois, hw, dim)

        return ret.flatten(1)
        

@MODELS.register_module()
class MaskGenerator2D_MixAttn(BaseModule):
    """2D Mask generator model"""
    def __init__(self, 
                 feat_dim: int = 64,  # feature dim
                 size: int = 7,  # width and height of ROI
                 layer_num: int = None,  # ommited
                 conv_layer_num: int = 2,
                 attn_layer_num: int = 1, 
                 norm: str = 'LayerNorm',  # norm layer
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)

        self.feat_dim = feat_dim
        self.input_size = size 

        self.conv_layer_num = conv_layer_num
        self.convs = nn.ModuleList()
        for layer_idx in range(conv_layer_num):
            # (num_roi, c, h, w) -> (num_roi, c // 2, h // 2, w // 2)
            self.convs.append(
                ConvModule(
                    in_channels=self.feat_dim, 
                    out_channels=self.feat_dim // 2 if layer_idx == conv_layer_num - 1 else self.feat_dim, 
                    kernel_size=3,
                    stride=2,
                    padding=1, 
                    norm_cfg=dict(type='GN', num_groups=16)
                )
            )

            self.input_size = (self.input_size + 1) // 2

        # update feat_dim and size
        self.feat_dim = self.feat_dim // 2

        self.attn_layer_num = attn_layer_num
        self.query, self.key, self.value = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.norm = nn.ModuleList()

        self.query_cross, self.key_cross, self.value_cross = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.norm_cross = nn.ModuleList()
        
        if norm == 'GroupNorm':
            norm_type = nn.GroupNorm
            params = dict(num_groups=32, num_channels=self.feat_dim)
        elif norm == 'LayerNorm':
            norm_type = nn.LayerNorm
            params = dict(normalized_shape=[self.feat_dim])
        else: raise NotImplementedError

        for layer_idx in range(attn_layer_num):
            self.query.append(
                nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim)
            )
            self.key.append(
                nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim)
            )
            self.value.append(
                nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim)
            )

            self.norm.append(norm_type(**params))

            self.query_cross.append(
                nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim)
            )
            self.key_cross.append(
                nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim)
            )
            self.value_cross.append(
                nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim)
            )

            self.norm_cross.append(norm_type(**params))

        self.final_fc1 = nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim)
        self.final_fc2 = nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim)
        self.act = nn.SiLU()

        self.idx_ = 0

    def forward_cross_attn(self, x: Tensor):
        
        x = x.permute(2, 0, 1)  # (hw, num_of_rois, dim)

        for layer_idx in range(self.attn_layer_num):
            # conv
            q = self.query_cross[layer_idx](x)  # (hw, num_of_rois, dim)
            k = self.key_cross[layer_idx](x)
            v = self.value_cross[layer_idx](x)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.feat_dim ** 0.5)  # (hw, num_of_rois, num_of_rois)
            attn_scores = F.softmax(attn_scores, dim=-1)  # (hw, num_of_rois, num_of_rois)

            x = x + torch.matmul(attn_scores, v)  # (hw, num_of_rois, dim)

            x = self.norm_cross[layer_idx](x)  # (hw, num_of_rois, dim)

        return torch.sigmoid(x.permute(1, 2, 0))
    
    def forward_self_attn(self, x: Tensor):
        x = x.permute(0, 2, 1)  # (num_of_rois, hw, dim)

        for layer_idx in range(self.attn_layer_num):
            # conv
            q = self.query[layer_idx](x)  # (num_of_rois, hw, dim)
            k = self.key[layer_idx](x)
            v = self.value[layer_idx](x)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.feat_dim ** 0.5)  # (num_of_rois, hw, hw)
            attn_scores = F.softmax(attn_scores, dim=-1)  # (num_of_rois, hw, hw)

            x = x + torch.matmul(attn_scores, v)  # (num_of_rois, hw, dim)

            x = self.norm[layer_idx](x)  # (num_of_rois, hw, dim)

        return torch.sigmoid(x.permute(0, 2, 1))
    
    def forward_mix(self, x: Tensor):
        x = x.permute(0, 2, 1)  # (num_of_rois, hw, dim)

        for layer_idx in range(self.attn_layer_num):

            ## Self
            q = self.query[layer_idx](x)  # (num_of_rois, hw, dim)
            k = self.key[layer_idx](x)
            v = self.value[layer_idx](x)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.feat_dim ** 0.5)  # (num_of_rois, hw, hw)
            attn_scores = F.softmax(attn_scores, dim=-1)  # (num_of_rois, hw, hw)

            x = x + torch.matmul(attn_scores, v)  # (num_of_rois, hw, dim)
            x = self.norm[layer_idx](x)  # (num_of_rois, hw, dim)
            
            ## Cross
            x = x.permute(1, 0, 2)  # (hw, num_of_rois, dim)
            q_ = self.query_cross[layer_idx](x)  # (hw, num_of_rois, dim)
            k_ = self.key_cross[layer_idx](x)
            v_ = self.value_cross[layer_idx](x)

            attn_scores = torch.matmul(q_, k_.transpose(-2, -1)) / (self.feat_dim ** 0.5)  # (hw, num_of_rois, num_of_rois)
            attn_scores = F.softmax(attn_scores, dim=-1)  # (hw, num_of_rois, num_of_rois)

            x = x + torch.matmul(attn_scores, v_)  # (hw, num_of_rois, dim)

            x = self.norm_cross[layer_idx](x)  # (hw, num_of_rois, dim)

            if (layer_idx < self.attn_layer_num - 1):  x = x.permute(1, 0, 2)

        x = self.final_fc1(x)

        return x
        
    def forward(self, x: Tensor, mode: str = 'mix'):
        # x: (num_of_rois, dim, h, w)

        for layer_idx in range(self.conv_layer_num):
            x = self.convs[layer_idx](x)

        x = x.flatten(2)  # (num_of_rois, dim, hw)
            
        if mode == 'separate':  # TODO 
            x_ = torch.clone(x)

            mask_self = self.forward_self_attn(x)
            mask_cross = self.forward_cross_attn(x_)

            mask = mask_self + mask_cross

            return mask
        
        else:
            x = self.forward_mix(x)
            x = x.permute(1, 2, 0)
            return x.flatten(1)
        

@MODELS.register_module()
class MaskGenerator2D_MixAttn2(BaseModule):
    """2D Mask generator model"""
    def __init__(self, 
                 feat_dim: int = 64,  # feature dim
                 size: int = 7,  # width and height of ROI
                 layer_num: int = None,  # ommited
                 conv_layer_num: int = 2,
                 attn_layer_num: int = 2, 
                 norm: str = 'LayerNorm',  # norm layer
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)

        self.feat_dim = feat_dim
        self.input_size = size 

        self.conv_layer_num = conv_layer_num
        self.convs = nn.ModuleList()
        for layer_idx in range(conv_layer_num):
            # (num_roi, c, h, w) -> (num_roi, c // 2, h // 2, w // 2)
            self.convs.append(
                ConvModule(
                    in_channels=self.feat_dim, 
                    out_channels=self.feat_dim // 2 if layer_idx == conv_layer_num - 1 else self.feat_dim, 
                    kernel_size=3,
                    stride=2,
                    padding=1, 
                    norm_cfg=dict(type='GN', num_groups=16)
                )
            )

            self.input_size = (self.input_size + 1) // 2

        # update feat_dim and size
        self.feat_dim = self.feat_dim // 2

        self.attn_layer_num = attn_layer_num
        self.self_attn, self.cross_attn = nn.ModuleList(), nn.ModuleList()

        for layer_idx in range(attn_layer_num):
            self.self_attn.append(
                nn.TransformerEncoderLayer(d_model=self.feat_dim, 
                                           nhead=1, 
                                           dim_feedforward=self.feat_dim, 
                                           batch_first=True, 
                                           )
            )

            self.cross_attn.append(
                nn.TransformerEncoderLayer(d_model=self.feat_dim, 
                                           nhead=1, 
                                           dim_feedforward=self.feat_dim, 
                                           batch_first=True)
            )

        self.final_fc = nn.Linear(in_features=self.feat_dim, out_features=self.feat_dim)

    def forward_mix(self, x: Tensor):
        x = x.permute(0, 2, 1)  # (num_of_rois, hw, dim)

        for layer_idx in range(self.attn_layer_num):

            x = self.self_attn[layer_idx](x)  # (num_of_rois, hw, dim)

            x = x.permute(1, 0, 2)  # (hw, num_of_rois, dim)
            x = self.cross_attn[layer_idx](x)

            if (layer_idx < self.attn_layer_num - 1):  x = x.permute(1, 0, 2)

        x = self.final_fc(x)

        return x
    
    def forward(self, x: Tensor, mode: str = 'mix'):
        for layer_idx in range(self.conv_layer_num):
            x = self.convs[layer_idx](x)

        x = x.flatten(2)  # (num_of_rois, dim, hw)
            
        if mode == 'separate':  # TODO 
            raise NotImplementedError
        
        else:
            x = self.forward_mix(x)
            x = x.permute(1, 2, 0)
            return x.flatten(1)
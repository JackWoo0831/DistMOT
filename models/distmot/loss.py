"""
Loss function for history memory
"""

from typing import Optional

import torch
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.losses.utils import reduce_loss, weight_reduce_loss

@MODELS.register_module()
class DistMOTLoss(BaseModule):
    def __init__(self, 
                 reduction: str = 'mean', 
                 loss_weight: float = 1.0, 
                 manner: str = 'entropy') -> None:
        super(DistMOTLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.manner = manner  # origin, sim, entropy

    def multi_pos_info_nce(self,
                           dot_product: Tensor,
                           label: Tensor,
                           cos_sim: Tensor = None, 
                           entropy: Tensor = None, 
                           weight: Optional[Tensor] = None,
                           reduction: str = 'mean',
                           avg_factor: Optional[float] = None) -> Tensor:
        
        pos_inds = (label >= 1)
        neg_inds = (label == 0)

        if self.manner == 'entropy':
            # l_q = \log [1 + w_i \sum_{k^+} \sum_{k^-} (e^- \dot (q k^-) - e^+ \dot (q k^+) )]
            entropy[pos_inds] = 1. - entropy[pos_inds]
            dot_product = dot_product * entropy 

        pred_pos = dot_product * pos_inds.float()
        pred_neg = dot_product * neg_inds.float()
        # use -inf to mask out unwanted elements.
        pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
        pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

        if weight is not None:
            weight = weight.float()

        _pos_expand = torch.repeat_interleave(pred_pos, dot_product.shape[1], dim=1)
        _neg_expand = pred_neg.repeat(1, dot_product.shape[1])

        x = torch.nn.functional.pad((_neg_expand - _pos_expand), (0, 1),
                                    'constant', 0)
        loss = torch.logsumexp(x, dim=1)      
        
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
            
        return loss

    def forward(self,
                dot_product: Tensor,
                label: Tensor,
                cos_sim: Tensor = None, 
                entropy: Tensor = None, 
                weight: Optional[Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:

        assert dot_product.size() == label.size()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        loss = self.loss_weight * self.multi_pos_info_nce(
            dot_product,
            label,
            cos_sim=cos_sim, 
            entropy=entropy, 
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss

"""
Sampler with recording tracking id
"""
from mmdet.models.task_modules.assigners import AssignResult
from mmdet.registry import TASK_UTILS

from mmdet.models.task_modules.samplers import BaseSampler, SamplingResult, CombinedSampler
from mmengine.structures import InstanceData
from torch import Tensor
import torch

from mmdet.structures.bbox import BaseBoxes, cat_boxes


class SamplingResult_track(SamplingResult):
    def __init__(self, pos_inds: Tensor,
                       neg_inds: Tensor,
                       priors: Tensor, 
                       gt_bboxes: Tensor, 
                       assign_result: AssignResult, 
                       gt_flags: Tensor, 
                       avg_factor_with_neg: bool = True) -> None:
        super().__init__(pos_inds, neg_inds, priors, gt_bboxes, assign_result, gt_flags, avg_factor_with_neg)

        self.pos_assigned_track_ids = assign_result._extra_properties['gt_track_ids'][pos_inds]

    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_priors': self.pos_priors,
            'neg_priors': self.neg_priors,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
            'pos_assigned_track_ids': self.pos_assigned_track_ids, 
            'num_pos': self.num_pos,
            'num_neg': self.num_neg,
            'avg_factor': self.avg_factor
        }
    
@TASK_UTILS.register_module()
class CombinedSampler_track(CombinedSampler):

    def sample(self, assign_result: AssignResult, pred_instances: InstanceData, gt_instances: InstanceData, **kwargs) -> SamplingResult:
        
        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors
        gt_labels = gt_instances.labels
        gt_instances_ids = gt_instances.instances_ids


        if len(priors.shape) < 2:
            priors = priors[None, :]

        gt_flags = priors.new_zeros((priors.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            # When `gt_bboxes` and `priors` are all box type, convert
            # `gt_bboxes` type to `priors` type.
            if (isinstance(gt_bboxes, BaseBoxes)
                    and isinstance(priors, BaseBoxes)):
                gt_bboxes_ = gt_bboxes.convert_to(type(priors))
            else:
                gt_bboxes_ = gt_bboxes
            priors = cat_boxes([gt_bboxes_, priors], dim=0)
            assign_result.add_gt_(gt_labels)  # add bboxes, ious, indexes, ... to gt
            
            # besides, we need add track ids
            assign_result._extra_properties['gt_track_ids'] = torch.cat([gt_instances_ids, assign_result._extra_properties['gt_track_ids']])

            assert assign_result._extra_properties['gt_track_ids'].shape[0] == assign_result.gt_inds.shape[0]

            gt_ones = priors.new_ones(gt_bboxes_.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=priors, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=priors, **kwargs)
        neg_inds = neg_inds.unique()

        sampling_result = SamplingResult_track(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=priors,
            gt_bboxes=gt_bboxes,
            assign_result=assign_result,
            gt_flags=gt_flags)
            
        return sampling_result
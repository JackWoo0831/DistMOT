"""
Max Iou Assigner with recording tracking id
"""
import copy
from typing import Optional, Union

import torch
from mmengine.structures import InstanceData
from torch import Tensor


from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules.assigners import AssignResult, MaxIoUAssigner

def _perm_box(bboxes,
              iou_calculator,
              iou_thr=0.97,
              perm_range=0.01,
              counter=0,
              max_iter=5):
    """Compute the permuted bboxes.

    Args:
        bboxes (Tensor): Shape (n, 4) for , "xyxy" format.
        iou_calculator (obj): Overlaps Calculator.
        iou_thr (float): The permuted bboxes should have IoU > iou_thr.
        perm_range (float): The scale of permutation.
        counter (int): Counter of permutation iteration.
        max_iter (int): The max iterations of permutation.
    Returns:
        Tensor: The permuted bboxes.
    """
    ori_bboxes = copy.deepcopy(bboxes)
    is_valid = True
    N = bboxes.size(0)
    perm_factor = bboxes.new_empty(N, 4).uniform_(1 - perm_range,
                                                  1 + perm_range)
    bboxes *= perm_factor
    new_wh = bboxes[:, 2:] - bboxes[:, :2]
    if (new_wh <= 0).any():
        is_valid = False
    iou = iou_calculator(ori_bboxes.unique(dim=0), bboxes)
    if (iou < iou_thr).any():
        is_valid = False
    if not is_valid and counter < max_iter:
        return _perm_box(
            ori_bboxes,
            iou_calculator,
            perm_range=max(perm_range - counter * 0.001, 1e-3),
            counter=counter + 1)
    return bboxes


def perm_repeat_bboxes(bboxes, iou_calculator=None, perm_repeat_cfg=None):
    """Permute the repeated bboxes.

    Args:
        bboxes (Tensor): Shape (n, 4) for , "xyxy" format.
        iou_calculator (obj): Overlaps Calculator.
        perm_repeat_cfg (Dict): Config of permutation.
    Returns:
        Tensor: Bboxes after permuted repeated bboxes.
    """
    assert isinstance(bboxes, torch.Tensor)
    if iou_calculator is None:
        import torchvision
        iou_calculator = torchvision.ops.box_iou
    bboxes = copy.deepcopy(bboxes)
    unique_bboxes = bboxes.unique(dim=0)
    iou_thr = perm_repeat_cfg.get('iou_thr', 0.97)
    perm_range = perm_repeat_cfg.get('perm_range', 0.01)
    for box in unique_bboxes:
        inds = (bboxes == box).sum(-1).float() == 4
        if inds.float().sum().item() == 1:
            continue
        bboxes[inds] = _perm_box(
            bboxes[inds],
            iou_calculator,
            iou_thr=iou_thr,
            perm_range=perm_range,
            counter=0)
    return bboxes

@TASK_UTILS.register_module()
class MaxIoUAssigner_track(MaxIoUAssigner):
    
    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        
        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors
        gt_labels = gt_instances.labels
        gt_instances_ids = gt_instances.instances_ids  

        if gt_instances_ignore is not None:
            gt_bboxes_ignore = gt_instances_ignore.bboxes
        else:
            gt_bboxes_ignore = None

        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False  #  False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = priors.device
            priors = priors.cpu()
            gt_bboxes = gt_bboxes.cpu()
            gt_labels = gt_labels.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()

        if self.perm_repeat_gt_cfg is not None and priors.numel() > 0:  # False 
            gt_bboxes_unique = perm_repeat_bboxes(gt_bboxes,
                                                  self.iou_calculator,
                                                  self.perm_repeat_gt_cfg)
        else:
            gt_bboxes_unique = gt_bboxes
        overlaps = self.iou_calculator(gt_bboxes_unique, priors)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and priors.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    priors, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, priors, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels, gt_instances_ids)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result
    
    def assign_wrt_overlaps(self, overlaps: Tensor,
                            gt_labels: Tensor, gt_instances_ids: Tensor) -> AssignResult:
        """
        Add extra 'gt_track_ids' to AssignResult
        """

        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        assigned_gt_instances = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            assigned_labels = overlaps.new_full((num_bboxes, ),
                                                -1,
                                                dtype=torch.long)
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
                assigned_gt_instances[:] = -1
            ret = AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=max_overlaps,
                labels=assigned_labels)
            ret.set_extra_property('gt_track_ids', assigned_gt_instances)

            return ret

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)  # (num_bboxes, )
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)  # (num_gts, )

        # 2. assign negative: below
        # the negative inds are set to be 0
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        assigned_gt_instances = gt_instances_ids[argmax_overlaps]
        assigned_gt_instances[pos_inds] = assigned_gt_instances[pos_inds]  # pos_inds: which gt match the prior best
        assigned_gt_instances[torch.logical_not(pos_inds)] = -1 

        assert assigned_gt_inds.shape[0] == assigned_gt_instances.shape[0]

        if self.match_low_quality:  # False
            # Low-quality matching will overwrite the assigned_gt_inds assigned
            # in Step 3. Thus, the assigned gt might not be the best one for
            # prediction.
            # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
            # bbox 1 will be assigned as the best target for bbox A in step 3.
            # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
            # assigned_gt_inds will be overwritten to be bbox 2.
            # This might be the reason that it is not used in ROI Heads.
            for i in range(num_gts):
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    if self.gt_max_assign_all:
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] -
                                                  1]

        ret = AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=max_overlaps,
            labels=assigned_labels)
        
        ret.set_extra_property('gt_track_ids', assigned_gt_instances)

        return ret
"""
Tracking head for distmot
"""
from typing import List, Optional

from mmengine.model import BaseModule
from torch import Tensor
import torch.nn as nn

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import TrackSampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import InstanceList
from mmdet.models.tracking_heads import QuasiDenseTrackHead

@MODELS.register_module()
class DistMOTHead(QuasiDenseTrackHead):
    
    def loss(self, key_feats: List[Tensor], ref_feats: List[Tensor],
             rpn_results_list: InstanceList,
             ref_rpn_results_list: InstanceList, data_samples: TrackSampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            key_feats (list[Tensor]): list of multi-level image features.
            ref_feats (list[Tensor]): list of multi-level ref_img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals of key img.
            ref_rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals of ref img.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict: A dictionary of loss components.
        """
        assert self.with_track
        num_imgs = len(data_samples)
        batch_gt_instances = []
        ref_batch_gt_instances = []
        batch_gt_instances_ignore = []
        gt_match_indices_list = []

        video_id_list = []  # mark video id, len == batch size

        for track_data_sample in data_samples:
            key_data_sample = track_data_sample.get_key_frames()[0]
            ref_data_sample = track_data_sample.get_ref_frames()[0]
            batch_gt_instances.append(key_data_sample.gt_instances)
            ref_batch_gt_instances.append(ref_data_sample.gt_instances)
            if 'ignored_instances' in key_data_sample:
                batch_gt_instances_ignore.append(
                    key_data_sample.ignored_instances)
            else:
                batch_gt_instances_ignore.append(None)
            # get gt_match_indices
            ins_ids = key_data_sample.gt_instances.instances_ids.tolist()
            ref_ins_ids = ref_data_sample.gt_instances.instances_ids.tolist()
            match_indices = Tensor([
                ref_ins_ids.index(i) if (i in ref_ins_ids and i > 0) else -1
                for i in ins_ids
            ]).to(key_feats[0].device)
            gt_match_indices_list.append(match_indices)

            video_id_list.append(key_data_sample.video_id)

        key_sampling_results, ref_sampling_results = [], []
        for i in range(num_imgs):
            rpn_results = rpn_results_list[i]
            ref_rpn_results = ref_rpn_results_list[i]
            # rename ref_rpn_results.bboxes to ref_rpn_results.priors
            ref_rpn_results.priors = ref_rpn_results.pop('bboxes')

            # assign rpn with gt
            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in key_feats])
            key_sampling_results.append(sampling_result)

            ref_assign_result = self.bbox_assigner.assign(
                ref_rpn_results, ref_batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            ref_sampling_result = self.bbox_sampler.sample(
                ref_assign_result,
                ref_rpn_results,
                ref_batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in ref_feats])
            ref_sampling_results.append(ref_sampling_result)

        key_bboxes = [res.pos_bboxes for res in key_sampling_results]
        key_roi_feats = self.extract_roi_feats(key_feats, key_bboxes)
        ref_bboxes = [res.bboxes for res in ref_sampling_results]
        ref_roi_feats = self.extract_roi_feats(ref_feats, ref_bboxes)

        """
        # make video_ids list and rois are one-to-one
        num_key_rois = [res.pos_bboxes.size(0) for res in key_sampling_results]
        video_id_list_ = []
        for idx, video_id in enumerate(video_id_list):
            video_id_list_.extend([video_id] * num_key_rois[idx]) 
        """

        loss_track = self.embed_head.loss(key_roi_feats, ref_roi_feats,
                                          key_sampling_results,
                                          ref_sampling_results,
                                          gt_match_indices_list,
                                          video_id_list)

        return loss_track
    
    def predict(self, feats: List[Tensor],
                rescaled_bboxes: List[Tensor]) -> Tensor:
        """Perform forward propagation of the tracking head and predict
        tracking results on the features of the upstream network.

        Args:
            feats (list[Tensor]): Multi level feature maps of `img`.
            rescaled_bboxes (list[Tensor]): list of rescaled bboxes in sampling
                result.

        Returns:
            Tensor: The extracted track features.
        """
        bbox_feats = self.extract_roi_feats(feats, rescaled_bboxes)
        track_feats = self.embed_head.predict(bbox_feats)
        return track_feats
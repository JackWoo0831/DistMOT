"""
DistMOT model, based on YOLOX.
"""

from typing import List, Tuple, Union, Optional

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import TrackSampleList, SampleList
from mmdet.utils import InstanceList, OptMultiConfig
from mmengine.config import ConfigDict
from mmdet.utils import OptConfigType, OptMultiConfig

from mmdet.models.mot.base import BaseMOTModel
from mmdet.models.dense_heads import YOLOXHead

@MODELS.register_module()  # set NMS=False in YOLOXHead
class YOLOXHead_woNMS(YOLOXHead):

    def loss_and_predict(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None, 
        with_nms: bool = True
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.
            proposal_cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.

        Returns:
            tuple: the return value is a tuple contains:

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - predictions (list[:obj:`InstanceData`]): Detection
                  results of each image after the post process.
        """
        outputs = self.unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(x)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, cfg=proposal_cfg, with_nms=with_nms)
        return losses, predictions
    
    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False, 
                with_nms: bool = True) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(x)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale, with_nms=with_nms)
        return predictions
    
    @staticmethod
    def unpack_gt_instances(batch_data_samples: SampleList) -> tuple:
        """Unpack ``gt_instances``, ``gt_instances_ignore`` and ``img_metas`` based
        on ``batch_data_samples``

        Args:
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple:

                - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                    gt_instance. It usually includes ``bboxes`` and ``labels``
                    attributes.
                - batch_gt_instances_ignore (list[:obj:`InstanceData`]):
                    Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                    data that is ignored during training and testing.
                    Defaults to None.
                - batch_img_metas (list[dict]): Meta information of each image,
                    e.g., image size, scaling factor, etc.
        """
        batch_gt_instances = []
        batch_gt_instances_ignore = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            if 'ignored_instances' in data_sample:
                batch_gt_instances_ignore.append(data_sample.ignored_instances)
            else:
                batch_gt_instances_ignore.append(None)

        return batch_gt_instances, batch_gt_instances_ignore, batch_img_metas


@MODELS.register_module()
class DistMOT(BaseMOTModel):
    """
    DistMOT based on YOLOX with a reid head
    The whole architecture follows QDTrack
    """
    
    def __init__(self,
                 detector: Optional[dict] = None,
                 track_head: Optional[dict] = None,
                 tracker: Optional[dict] = None,
                 freeze_detector: bool = False,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)
        if detector is not None:
            self.detector = MODELS.build(detector)

        if track_head is not None:
            self.track_head = MODELS.build(track_head)

        if tracker is not None:
            self.tracker = MODELS.build(tracker)

        self.freeze_detector = freeze_detector
        if self.freeze_detector:
            self.freeze_module('detector')
        

    def predict(self,
                inputs: Tensor,
                data_samples: TrackSampleList,
                rescale: bool = True,
                **kwargs) -> TrackSampleList:
        """Predict results from a video and data samples with post- processing.

        Args:
            inputs (Tensor): of shape (N, T, C, H, W) encoding
                input images. The N denotes batch size.
                The T denotes the number of frames in a video.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `video_data_samples`.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to True.

        Returns:
            TrackSampleList: Tracking results of the inputs.
        """

        assert inputs.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert inputs.size(0) == 1, \
            'QDTrack inference only support 1 batch size per gpu for now.'

        assert len(data_samples) == 1, \
            'QDTrack only support 1 batch size per gpu for now.'

        track_data_sample = data_samples[0]
        video_len = len(track_data_sample)
        if track_data_sample[0].frame_id == 0:
            self.tracker.reset()

        for frame_id in range(video_len):
            img_data_sample = track_data_sample[frame_id]
            single_img = inputs[:, frame_id].contiguous()
            x = self.detector.extract_feat(single_img)

            det_results = self.detector.bbox_head.predict(
                x, [img_data_sample], rescale=rescale
            )

            assert len(det_results) == 1, 'Batch inference is not supported.'
            img_data_sample.pred_instances = det_results[0]
            frame_pred_track_instances = self.tracker.track(
                model=self,
                img=single_img,
                feats=x,
                data_sample=img_data_sample,
                **kwargs)
            img_data_sample.pred_track_instances = frame_pred_track_instances

        return [track_data_sample]
        
        
    # TODO
    def loss(self, inputs: Tensor, data_samples: TrackSampleList,
             **kwargs) -> Union[dict, tuple]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Dict[str, Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size. The T denotes the number of
                frames.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `video_data_samples`.

        Returns:
            dict: A dictionary of loss components.
        """
        assert inputs.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert inputs.size(1) == 2, \
            'QDTrack can only have 1 key frame and 1 reference frame.'
        

        # split the data_samples into two aspects: key frames and reference
        # frames
        ref_data_samples, key_data_samples = [], []
        key_frame_inds, ref_frame_inds = [], []
        # set cat_id of gt_labels to 0 in RPN
        for track_data_sample in data_samples:
            key_frame_inds.append(track_data_sample.key_frames_inds[0])
            ref_frame_inds.append(track_data_sample.ref_frames_inds[0])
            key_data_sample = track_data_sample.get_key_frames()[0]
            # key_data_sample.gt_instances.labels = \
            #     torch.zeros_like(key_data_sample.gt_instances.labels)
            key_data_samples.append(key_data_sample)
            ref_data_sample = track_data_sample.get_ref_frames()[0]
            ref_data_samples.append(ref_data_sample)

        key_frame_inds = torch.tensor(key_frame_inds, dtype=torch.int64)
        ref_frame_inds = torch.tensor(ref_frame_inds, dtype=torch.int64)
        batch_inds = torch.arange(len(inputs))
        key_imgs = inputs[batch_inds, key_frame_inds].contiguous()
        ref_imgs = inputs[batch_inds, ref_frame_inds].contiguous()

        x = self.detector.extract_feat(key_imgs) 

        ref_x = self.detector.extract_feat(ref_imgs)

        losses = dict()

        bbox_loss, bbox_results_list = self.detector.bbox_head.loss_and_predict(x, 
                                                                      key_data_samples, 
                                                                      proposal_cfg=None,
                                                                      with_nms=True, )  # TODO proposal_cfg
        
        ref_bbox_results_list = self.detector.bbox_head.predict(ref_x, 
                                                               ref_data_samples, 
                                                               with_nms=True,
                                                               **kwargs)

        losses.update(bbox_loss)

        # NOTE: For debug
        # print(bbox_results_list[0].scores.shape[0], ref_bbox_results_list[0].scores.shape[0])

        # NOTE: add 'priors' attribute to fit MaxIouAssiger
        for res in bbox_results_list: res.priors = res.bboxes
        for res in ref_bbox_results_list: res.priors = res.bboxes        

        # tracking head loss
        losses_track = self.track_head.loss(x, ref_x, bbox_results_list,
                                            ref_bbox_results_list, data_samples,
                                            **kwargs)
        losses.update(losses_track)

        return losses
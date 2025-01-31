"""
DistMOT Tracker
"""

from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import TrackDataSample
from mmdet.structures.bbox import bbox_overlaps
from mmdet.models.trackers.base_tracker import BaseTracker

import lap 

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np 

@MODELS.register_module()
class DistMOTTracker(BaseTracker):
    """
    
    """

    def __init__(self, 
                 init_score_thr: float = 0.8,
                 obj_score_thr: float = 0.5,
                 match_score_thr: float = 0.5,
                 memo_tracklet_frames: int = 10,
                 memo_backdrop_frames: int = 1,

                 memo_momentum: float = 0.8,  # removed

                 nms_conf_thr: float = 0.5,
                 nms_backdrop_iou_thr: float = 0.3,
                 nms_class_iou_thr: float = 0.7,
                 with_cats: bool = True,
                 match_metric: str = 'bisoftmax',

                 # memory bank 
                 append_sim_thr: float = 0.9,  # for o_j^t, if forall o_j^{t'} (t' < t): sim(o_j^t, o_j^{t'}) < thr, -> push_back o_j^t
                 max_buffer_size: int = 10,  # maximum store buffer for object, if exceeded, remove first-in

                 # solver
                 solver: str = 'greedy',  # solver, 'greedy' or 'optimal'
                 
                 **kwargs) -> None:
        super().__init__(**kwargs)

        assert memo_tracklet_frames >= 0
        assert memo_backdrop_frames >= 0
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.memo_tracklet_frames = memo_tracklet_frames
        self.memo_backdrop_frames = memo_backdrop_frames

        self.nms_conf_thr = nms_conf_thr
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.nms_class_iou_thr = nms_class_iou_thr
        self.with_cats = with_cats
        assert match_metric in ['bisoftmax', 'softmax', 'cosine', 'dotproduct']
        self.match_metric = match_metric

        self.num_tracks = 0
        self.tracks = dict()
        self.backdrops = []

        # memory bank
        self.memo_bank = dict()  # {obj_id: Dict | InstanceData}
        self.append_sim_thr = append_sim_thr
        self.max_buffer_size = max_buffer_size

        # for debug
        self.frame_cnt = 0

        self.solver = solver

    def reset(self):
        """Reset the buffer of the tracker."""
        self.num_tracks = 0
        self.tracks = dict()
        self.backdrops = []
        self.memo_bank = dict()


    def update(self, ids: Tensor, bboxes: Tensor, embeds: Tensor,
               labels: Tensor, scores: Tensor, frame_id: int) -> None:  
        """Update memory bank and tracks by current results"""
        tracklet_inds = ids > -1

        for id, bbox, embed, label, score in zip(ids[tracklet_inds],
                                                 bboxes[tracklet_inds],
                                                 embeds[tracklet_inds],
                                                 labels[tracklet_inds],
                                                 scores[tracklet_inds]):
            id = int(id)
            # update the tracked ones and initialize new tracks
            if id in self.tracks.keys():
                velocity = (bbox - self.tracks[id]['bbox']) / (
                    frame_id - self.tracks[id]['last_frame'])
                self.tracks[id]['bbox'] = bbox
                self.tracks[id]['embed'] = embed
                self.tracks[id]['last_frame'] = frame_id
                self.tracks[id]['label'] = label
                self.tracks[id]['score'] = score
                self.tracks[id]['velocity'] = (
                    self.tracks[id]['velocity'] * self.tracks[id]['acc_frame']
                    + velocity) / (
                        self.tracks[id]['acc_frame'] + 1)
                self.tracks[id]['acc_frame'] += 1
            else:
                self.tracks[id] = dict(
                    bbox=bbox,
                    embed=embed,
                    label=label,
                    score=score,
                    last_frame=frame_id,
                    velocity=torch.zeros_like(bbox),
                    acc_frame=0)
                
            # update memory bank of id-th obj
            self.udpate_memo_bank(id, embed, frame_id)

        # backdrop update according to IoU
        backdrop_inds = torch.nonzero(ids == -1, as_tuple=False).squeeze(1)
        ious = bbox_overlaps(bboxes[backdrop_inds], bboxes)
        for i, ind in enumerate(backdrop_inds):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_inds[i] = -1
        backdrop_inds = backdrop_inds[backdrop_inds > -1]
        # old backdrops would be removed at first
        self.backdrops.insert(
            0,
            dict(
                bboxes=bboxes[backdrop_inds],
                embeds=embeds[backdrop_inds],
                labels=labels[backdrop_inds]))

        # pop memo
        invalid_ids = []
        for k, v in self.tracks.items():
            if frame_id - v['last_frame'] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

        if len(self.backdrops) > self.memo_backdrop_frames:
            self.backdrops.pop()

    
    def udpate_memo_bank(self, id: Union[Tensor, int], embed: Tensor, frame_id: int):
        """ Update the memory bank for id-th obj
        
        """
        if isinstance(id, Tensor): id = id.item()

        if id not in self.memo_bank.keys():
            # initialize a new id
            new_memo = dict()
            new_memo['embed'] = embed[None, ]
            new_memo['frame_id'] = [frame_id]
            self.memo_bank[id] = new_memo

        else:
            # cal the most similar one
            # embed: (dim, ) memo.embed: (num_of_storage, dim)
            cur_id_memo = self.memo_bank[id]['embed']
            similarities = torch.matmul(F.normalize(cur_id_memo, p=2, dim=1), 
                                        F.normalize(embed, p=2, dim=0))  # (num_of_storage, )
            
            if 0.4 < similarities.max() < self.append_sim_thr:
                cur_id_memo = \
                    torch.vstack([cur_id_memo, embed])
                        
            else:  # TODO find a stragegy like moving average
                pass

            # pop out
            if cur_id_memo.shape[0] > self.max_buffer_size:
                cur_id_memo = cur_id_memo[-self.max_buffer_size: ]

            self.memo_bank[id]['frame_id'].append(frame_id)
            self.memo_bank[id]['embed'] = cur_id_memo


    @property
    def memo(self) -> Tuple[Tensor, ...]:
        """Get tracks memory."""
        memo_embeds = []
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        # velocity of tracks
        memo_vs = []
        # get tracks
        for k, v in self.tracks.items():
            memo_bboxes.append(v['bbox'][None, :])
            memo_embeds.append(v['embed'][None, :])
            memo_ids.append(k)
            memo_labels.append(v['label'].view(1, 1))
            memo_vs.append(v['velocity'][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)

        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        memo_vs = torch.cat(memo_vs, dim=0)
        return memo_bboxes, memo_labels, memo_embeds, memo_ids.squeeze(
            0), memo_vs

    def track(self,
              model: torch.nn.Module,
              img: torch.Tensor,
              feats: List[torch.Tensor],
              data_sample: TrackDataSample,
              rescale=True,
              **kwargs) -> InstanceData:
        """ Tracking forward function.
        
        """
        metainfo = data_sample.metainfo
        bboxes = data_sample.pred_instances.bboxes
        labels = data_sample.pred_instances.labels
        scores = data_sample.pred_instances.scores

        self.frame_cnt += 1

        frame_id = metainfo.get('frame_id', -1)
        # create pred_track_instances
        pred_track_instances = InstanceData()

        # return zero bboxes if there is no track targets
        if bboxes.shape[0] == 0:
            ids = torch.zeros_like(labels)
            pred_track_instances = data_sample.pred_instances.clone()
            pred_track_instances.instances_id = ids
            return pred_track_instances

        # get track feats
        rescaled_bboxes = bboxes.clone()
        if rescale:
            scale_factor = rescaled_bboxes.new_tensor(
                metainfo['scale_factor']).repeat((1, 2))
            rescaled_bboxes = rescaled_bboxes * scale_factor
        track_feats = model.track_head.predict(feats, [rescaled_bboxes])
        # sort according to the object_score
        _, inds = scores.sort(descending=True)
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]
        embeds = track_feats[inds, :]

        # duplicate removal for potential backdrops and cross classes
        valids = bboxes.new_ones((bboxes.size(0)))
        ious = bbox_overlaps(bboxes, bboxes)
        for i in range(1, bboxes.size(0)):
            thr = self.nms_backdrop_iou_thr if scores[
                i] < self.obj_score_thr else self.nms_class_iou_thr
            if (ious[i, :i] > thr).any():
                valids[i] = 0
        valids = valids == 1
        bboxes = bboxes[valids]
        scores = scores[valids]
        labels = labels[valids]
        embeds = embeds[valids, :]

        # init ids container
        ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)

        # match if buffer is not empty
        # NOTE: omit backdrop temporarily
        if bboxes.size(0) > 0 and not self.empty:
            (memo_bboxes, memo_labels, memo_embeds, memo_ids,
             memo_vs) = self.memo  # get latest tracks
            
            # calculate iou matrix
            # iou_matrix = self.get_iou_matrix(bboxes, memo_bboxes)

            # calculate the most-like and most-dislike similarity matrix
            latest_like, most_like, most_dislike = self.get_similarity_matrix(embeds, memo_embeds)

            match_scores = latest_like + 0.1 * most_dislike
            
            d2t = match_scores.softmax(dim=1)
            t2d = match_scores.softmax(dim=0)
            match_scores = 0.5 * (d2t + t2d)

            """      
            print(f'------------{frame_id}------------')
            _, inds = torch.max(latest_like, dim=1)
            inds = torch.cat([torch.arange(0, latest_like.shape[0]).to(inds.device).view(-1, 1), inds.view(-1, 1)], dim=1)
            print('corresponding most_dislike', most_dislike[inds[:, 0], inds[:, 1]])
            print('corresponding most_like', most_like[inds[:, 0], inds[:, 1]])
            # print('most_dislike', most_dislike)
            # print('most_like', most_like)
            # print('latest_like', latest_like)
            print('match_scores', match_scores)
            if self.frame_cnt == 10: exit()
            """
                       

            if self.solver == 'greedy':
            # associate according to match_scores, qd-track manner
                for i in range(bboxes.size(0)):
                    conf, memo_ind = torch.max(match_scores[i, :], dim=0)
                    id = memo_ids[memo_ind]
                    if conf > self.match_score_thr:
                        if id > -1:
                            # keep bboxes with high object score
                            # and remove background bboxes
                            if scores[i] > self.obj_score_thr:
                                ids[i] = id
                                match_scores[:i, memo_ind] = 0
                                match_scores[i + 1:, memo_ind] = 0
                            else:
                                if conf > self.nms_conf_thr:
                                    ids[i] = -2
            else:
                match_scores = 1. - match_scores
                _, row, col = lap.lapjv(
                    match_scores.cpu().numpy(), extend_cost=True, cost_limit=0.8, 
                )

                for i in range(bboxes.shape[0]):
                    if row[i] > -1 and scores[i] > self.obj_score_thr: 
                        id = memo_ids[row[i]]
                        ids[i] = id

        # tSNE 
        # self._t_SNE_vis(embeds, id=ids)

        # initialize new tracks
        new_inds = (ids == -1) & (scores > self.init_score_thr).cpu()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracks, self.num_tracks + num_news, dtype=torch.long)
        self.num_tracks += num_news

        self.update(ids, bboxes, embeds, labels, scores, frame_id)
        tracklet_inds = ids > -1
        # update pred_track_instances
        pred_track_instances.bboxes = bboxes[tracklet_inds]
        pred_track_instances.labels = labels[tracklet_inds]
        pred_track_instances.scores = scores[tracklet_inds]
        pred_track_instances.instances_id = ids[tracklet_inds]

        return pred_track_instances
    
    def get_iou_matrix(self, detections: Tensor, trajectories: Tensor):
        """get iou matrix"""

        return bbox_overlaps(detections, trajectories)  # (num_of_dets, num_of_trajs)
    
    def get_similarity_matrix(self, det_embeds: Tensor, traj_embeds: Tensor, 
                              post_process: bool = False, ):
        """get similarity matrix from memory bank (latest, most_like and most_dislike)

        Args:
            det_embeds: torch.Tensor, (num of dets, dim)
            traj_embeds: torch.Tensor, (len(self.tracks.keys()), dim)
            post_process: bool, whether post process most-like and most-dislike
        
        """
        # for debug
        assert len(self.tracks.keys()) <= len(self.memo_bank.keys()), \
            f'length of tracks and memory banks should be less or equal, but got {len(self.tracks.keys()), len(self.memo_bank.keys())}'

        most_like = torch.zeros(
            (det_embeds.shape[0], len(self.tracks.keys())), 
            device=det_embeds.device)
        most_dislike = torch.zeros(
            (det_embeds.shape[0], len(self.tracks.keys())), 
            device=det_embeds.device)

        # norm
        det_embeds_ = det_embeds

        # NOTE reduce time complexity
        for det_idx in range(det_embeds_.shape[0]):

            for trk_idx, id in enumerate(self.tracks.keys()):
                similarities = torch.matmul(self.memo_bank[id]['embed'], det_embeds_[det_idx])  # (num of storage, )

                most_like[det_idx, trk_idx] = similarities.max()
                most_dislike[det_idx, trk_idx] = similarities.min()

        # latest similarity, use bi-softmax
        if self.match_metric == 'bisoftmax':
            latest_like = torch.matmul(det_embeds, traj_embeds.t())
            d2t_scores = latest_like.softmax(dim=1)
            t2d_scores = latest_like.softmax(dim=0)
            latest_like = 0.5 * (d2t_scores + t2d_scores)
        elif self.match_metric == 'cosine':
            latest_like = torch.matmul(
                det_embeds_, 
                F.normalize(traj_embeds, p=2, dim=1).t(), 
            )
        elif self.match_metric == 'dotproduct':
            latest_like = torch.matmul(det_embeds, traj_embeds.t())
        else: raise NotImplementedError

        if post_process:
            most_dislike[most_dislike > 0.3] = 0  # mostly-like same id, set 0
            most_dislike = -1 * most_dislike.abs()  # all set to negative

            # most_like[most_like < 0.5] = 0  # different id, set 0

        return latest_like, most_like, most_dislike

    def _t_SNE_vis(self, embeds: Tensor, id: Tensor, draw_3d: bool = False):
        """
        t-SNE visualization

        embeds: (N, feat_dim)
        """

        # draw by tracking id
        """
        assert embeds.shape[0] == id.shape[0]
        invalid = id == -1
        embeds = embeds[torch.logical_not(invalid)]
        id = id[torch.logical_not(invalid)]

        if invalid.all() or invalid.sum() < 5: return 
        """

        # draw by single embedding slice
        num_of_group = embeds.shape[1] // 32
        embeds_ = torch.chunk(embeds, chunks=num_of_group, dim=1)
        embeds_ = torch.cat(embeds_, dim=0)

        id = torch.cat([torch.Tensor([idx] * num_of_group) for idx in range(embeds.shape[0])], dim=0)

        if not draw_3d:
            tsne = TSNE(n_components=2, perplexity=min(30.0, max(5.0, float(embeds_.shape[0] - 10))))
            X_tsne = tsne.fit_transform(embeds_.cpu().numpy())

            plt.clf()
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=id.cpu().numpy())
            plt.title('t-SNE Visualization of Object Features')
            plt.savefig(f'./qdtrack_baseline_vis/tSNE_{self.frame_cnt}.jpg')
        else:
            tsne = TSNE(n_components=3, )
            X_tsne = tsne.fit_transform(embeds.cpu().numpy())

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2])
            fig.savefig(f'./qdtrack_baseline_vis/tSNE_{self.frame_cnt}.jpg')

    def _response_map_vis(self, track_embed: Tensor, det_embed: Tensor, id: int):
        """
        draw a response map

        """
        track_embed_ = torch.tensor(track_embed).view(16, 12)
        det_embed_ = torch.tensor(det_embed).view(16, 12)

        track_embed_ = F.interpolate(track_embed_.unsqueeze(0).unsqueeze(0), size=(32, 24), mode='bilinear')
        det_embed_ = F.interpolate(det_embed_.unsqueeze(0).unsqueeze(0), size=(32, 24), mode='bilinear')

        response_map = track_embed_ * det_embed_ 
        response_map = response_map.squeeze()

        response_map = response_map.cpu().numpy()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(response_map, cmap='hot', interpolation='nearest')
        # fig.colorbar()
        fig.savefig(f'./qdtrack_baseline_vis/resp_map_{self.frame_cnt}_{id}.jpg')



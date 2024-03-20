"""
implenmentation of memory bank
"""

import torch 
from torch import Tensor
import torch.nn.functional as F
from typing import List, Optional, Tuple
from mmdet.models.task_modules import SamplingResult

class MemoBank:
    def __init__(self, init_cfg: dict) -> None:
        
        self.max_size = init_cfg['max_size']
        self.seq_num = init_cfg['seq_num']  # sequence nums in train set

        self.memo_bank = dict()  # key: video_id, value: dict[instances_id: tensor]
        self.visited_videos = set()  # label visited videos

        # init memo bank
        for vid in range(1, self.seq_num + 1):
            self.memo_bank[vid] = dict()

    def check_new_videos(self, video_ids: list):
        # assert len(set(video_ids)) == 1  # TODO

        if not video_ids[0] in self.visited_videos:
            # new video begin
            self.memo_bank.clear()
            self.visited_videos.add(video_ids[0])

    def push(self, feats: Tensor, ref_feats: Tensor, 
             key_sampling_result: SamplingResult = None, ref_sampling_result: SamplingResult = None,
             video_id: int = 0, mode: str = 'mean') -> None:
        """push feats into memo bank
        
        Args:
        mode: store every roi (every) or average rois belonging to same object (mean)
        
        """
        feats = feats.detach()
        ref_feats = ref_feats.detach()
       
        id_list = key_sampling_result.pos_assigned_track_ids
        ref_id_list = ref_sampling_result.pos_assigned_track_ids

        assert feats.shape[0] == id_list.shape[0]

        if mode == 'every':
            for idx in range(id_list.shape[0]):
                instance_id = id_list[idx].item()
                
                if not instance_id in self.memo_bank[video_id]:
                    self.memo_bank[video_id][instance_id] = [feats[idx]]
                else:
                    # TODO push back rules
                    self.memo_bank[video_id][instance_id].append(feats[idx])

                # check max size
                if (len(self.memo_bank[video_id][instance_id]) > self.max_size):
                    self.memo_bank[video_id][instance_id] = self.memo_bank[video_id][instance_id][-self.max_size: ]

        elif mode == 'mean':

            instance_id_map = dict()  # obj_id -> List[idx]
            for idx in range(id_list.shape[0]):
                instance_id = id_list[idx].item()

                if not instance_id in instance_id_map.keys():
                    instance_id_map[instance_id] = [idx]
                else:
                    instance_id_map[instance_id].append(idx)

            for instance_id, idxes in instance_id_map.items():

                if not instance_id in self.memo_bank[video_id]:
                    self.memo_bank[video_id][instance_id] = [feats[idxes].mean(dim=0)]
                else:
                    # TODO push back rules
                    feat_candidate = feats[idxes].mean(dim=0)
                    if self._check_whether_push(video_id, instance_id, feat_candidate, ):
                        self.memo_bank[video_id][instance_id].append(feats[idxes].mean(dim=0))
        
                # check max size
                if (len(self.memo_bank[video_id][instance_id]) > self.max_size):
                    self.memo_bank[video_id][instance_id] = self.memo_bank[video_id][instance_id][-self.max_size: ]

    def gen_logits(self, key_roi_feats: Tensor, ref_roi_feats: Tensor,
             key_sampling_results: List[SamplingResult],
             ref_sampling_results: List[SamplingResult],
             video_id_list: List[int]):
        """
        parse all batch to single batch
        """
        num_key_rois = [res.pos_bboxes.size(0) for res in key_sampling_results]
        key_embeds = torch.split(key_roi_feats, num_key_rois)
        num_ref_rois = [res.bboxes.size(0) for res in ref_sampling_results]
        ref_embeds = torch.split(ref_roi_feats, num_ref_rois)

        dists, labels, weights = [], [], []

        batch_size = len(num_key_rois)
        assert len(video_id_list) == batch_size

        for batch_idx in range(batch_size):
            dist, label, weight = self.gen_logits_single(
                key_embeds[batch_idx], 
                ref_embeds[batch_idx], 
                key_sampling_result=key_sampling_results[batch_idx], 
                ref_sampling_result=ref_sampling_results[batch_idx], 
                video_id=video_id_list[batch_idx]
            )

            dists.append(dist)
            labels.append(label)
            weights.append(weight)

        return dists, labels, weights


    def gen_logits_single(self, feats: Tensor, ref_feats: Tensor, 
             key_sampling_result: SamplingResult = None, ref_sampling_result: SamplingResult = None,
             video_id: int = None):
        """
        gen dists and labels for a single batch
        """
        
        id_list = key_sampling_result.pos_assigned_track_ids
        assert feats.shape[0] == id_list.shape[0]

        # gen labels (targets): matrix, row: cur id, col: hist id
        history_ids = []
        history_feats = []

        # print(video_id, self.memo_bank[video_id].keys())  # for debug
        for id in self.memo_bank[video_id].keys():
            
            # For debug: simplify
            if id not in id_list: continue
            
            num_of_feats = len(self.memo_bank[video_id][id])
            history_ids.extend([id] * num_of_feats)
            history_feats.extend([f for f in self.memo_bank[video_id][id]])

        assert len(history_feats) == len(history_ids)

        # check if empty
        if not len(history_feats):
            self.push(feats, ref_feats, key_sampling_result, ref_sampling_result, video_id=video_id)
            return None, None, None 

        history_ids = torch.tensor(history_ids).to(feats.device)  # (num of feats, )
        history_feats = torch.vstack(history_feats, ).to(feats.device)  # (num of feats, dim)

        labels = id_list.view(-1, 1) == history_ids.view(1, -1)
        labels = labels.int()      
        weights = (labels.sum(dim=1) > 0).float()

        # cal sim between key samples and history memos
        # print(history_feats.shape)  # for debug
        dist = embed_similarity(feats, history_feats, )

        # For debug
        """
        max_values, max_indices = dist.max(dim=1)
        acc = 0
        for row in range(labels.shape[0]):
            if torch.sum(labels[row]) > 1: print('****')
            acc += labels[row][max_indices[row]] == 1
        print('acc: ', acc / labels.shape[0])
        """

        # push current key samples
        self.push(feats, ref_feats, key_sampling_result, ref_sampling_result, video_id=video_id)

        return dist, labels, weights
    
    def cal_history_loss(self, dists: Tensor, labels: Tensor):
        """
        calulate loss wrt history memeory

        Args:
            dists: shape (num_of_rois, num_of_memos)
            labels: shape (num_of_rois, num_of_memos)
        """

    def _check_whether_push(self, video_id: int, instance_id: int, feat: Tensor, 
                            enable: bool = True, 
                            push_high_thresh: float = 0.9, push_low_thresh: float = 0.0) -> bool:
        """
        check if the current feature need to push in the memo bank
        """
        if not enable: return True

        histoty_feats = self.memo_bank[video_id][instance_id]  # (num of memos, feat_dim), 1 <= num of memos <= size
        histoty_feats = torch.vstack(histoty_feats)

        # assert feat.shape[0] == 1

        if feat.ndim == 1: feat = feat[None, :]
        cos_sim = embed_similarity(feat, histoty_feats, method='cosine')

        if cos_sim.max() < push_high_thresh and cos_sim.min() > push_low_thresh:
            return True
        
        return False
    
    def _label_smoothing(self, label: Tensor):
        """
        label smoothing
        """



def embed_similarity(key_embeds: Tensor,
                     ref_embeds: Tensor,
                     method: str = 'dot_product',
                     temperature: int = -1) -> Tensor:
    """Calculate feature similarity from embeddings.

    Args:
        key_embeds (Tensor): Shape (N1, C).
        ref_embeds (Tensor): Shape (N2, C).
        method (str, optional): Method to calculate the similarity,
            options are 'dot_product' and 'cosine'. Defaults to
            'dot_product'.
        temperature (int, optional): Softmax temperature. Defaults to -1.

    Returns:
        Tensor: Similarity matrix of shape (N1, N2).
    """
    assert method in ['dot_product', 'cosine']

    if method == 'cosine':
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)

    similarity = torch.mm(key_embeds, ref_embeds.T)

    if temperature > 0:
        similarity /= float(temperature)
    return similarity
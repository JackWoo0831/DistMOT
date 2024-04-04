"""
implenmentation of memory bank
"""

import math
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

    def push(self, feats: Tensor, 
             key_sampling_result: SamplingResult = None, 
             cos_sim: Tensor = None, 
             entropy: Tensor = None, 
             history_ids: Tensor = None, 
             video_id: int = 0, mode: str = 'mean') -> None:
        """push feats into memo bank
        
        Args:
        mode: store every roi (every) or average rois belonging to same object (mean)
        
        """
        feats = feats.detach()
       
        id_list = key_sampling_result.pos_assigned_track_ids

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
                    # if self._check_whether_push(video_id, instance_id, feat_candidate, ):
                    if self._check_whether_push_2(cos_sim, entropy, idxes, history_ids == instance_id):
                        self.memo_bank[video_id][instance_id].append(feat_candidate)
        
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
        cos_dists, entropies = [], []

        batch_size = len(num_key_rois)
        assert len(video_id_list) == batch_size

        for batch_idx in range(batch_size):
            dist, label, weight, cos_dist, entropy = self.gen_logits_single(
                key_embeds[batch_idx], 
                ref_embeds[batch_idx], 
                key_sampling_result=key_sampling_results[batch_idx], 
                ref_sampling_result=ref_sampling_results[batch_idx], 
                video_id=video_id_list[batch_idx]
            )

            dists.append(dist)
            cos_dists.append(cos_dist)
            labels.append(label)
            weights.append(weight)
            entropies.append(entropy)

        return dists, labels, weights, cos_dists, entropies


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
            self.push(feats, key_sampling_result, video_id=video_id)
            return None, None, None, None, None

        history_ids = torch.tensor(history_ids).to(feats.device)  # (num of feats, )
        history_feats = torch.vstack(history_feats, ).to(feats.device)  # (num of feats, dim)

        labels = id_list.view(-1, 1) == history_ids.view(1, -1)
        labels = labels.int()      
        weights = (labels.sum(dim=1) > 0).float()

        # cal sim between key samples and history memos
        # print(history_feats.shape)  # for debug
        dist = embed_similarity(feats, history_feats, )
        cos_dist = embed_similarity(feats.detach(), history_feats, method='cosine')  # NOTE omit grad
        assert not cos_dist.requires_grad

        entropy = self.cal_shannon_entropy(cos_dist, history_ids)  # (num of cur, num of hist)

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
        self.push(feats, key_sampling_result, cos_dist, entropy, history_ids, video_id=video_id)

        return dist, labels, weights, cos_dist, entropy
    
    def cal_shannon_entropy(self, cos_sim: Tensor, history_ids: Tensor) -> Tensor:
        """
        cal shannon entropy along dim 1

        Args:
            cos_dim: shape (num of cur feats, num of memo of corresponding id)
            history_ids: shape (num of memo feat of corresponding id, )
        """

        hist_ids_unique = torch.unique(history_ids)
        ret = torch.zeros_like(cos_sim)

        for id in hist_ids_unique:
            col_idxes = history_ids == id  # if col_idxes.shape[1] == 1, entropy = 0

            # E = \sum_i H(i) / N, H(i) = p \log p + (1 - p) \log (1 - p), N = num of hist memo

            cos_sim_region = cos_sim[:, col_idxes]
            cos_sim_region = torch.clamp(cos_sim_region, min=0.1, max=0.9)
            entropy = cos_sim_region * torch.log2(cos_sim_region) + \
                        (1 - cos_sim_region) * torch.log2(1 - cos_sim_region)
            
            assert not torch.isnan(entropy).any()  # TODO 

            entropy = torch.mean(-1 * entropy, dim=1, keepdim=True)
            ret[:, col_idxes] = entropy.repeat((1, cos_sim_region.shape[1]))

        return ret


    def _check_whether_push(self, video_id: int, instance_id: int, feat: Tensor, 
                            enable: bool = True, rule: str = 'sim',  # 'sim' or 'entropy'
                            sim_thresh: list = [0.2, 0.9], entropy_thresh: float = 0.5) -> bool:
        """
        check if the current feature need to push in the memo bank
        """
        if not enable: return True

        histoty_feats = self.memo_bank[video_id][instance_id]  # (num of memos, feat_dim), 1 <= num of memos <= size
        histoty_feats = torch.vstack(histoty_feats)

        # assert feat.shape[0] == 1

        if feat.ndim == 1: feat = feat[None, :]
        cos_sim = embed_similarity(feat, histoty_feats, method='cosine')  # (num of cur, num of memos)

        if rule == 'sim':

            if cos_sim.max() < sim_thresh[1] and cos_sim.min() > sim_thresh[0]:
                return True
            
        elif rule == 'entropy':
            sim_softmax = F.softmax(cos_sim, dim=1)  # (num of cur, num of memos)
            sim_log_softmax = F.log_softmax(cos_sim, dim=1)
            entropy = -1 * torch.sum(sim_softmax * sim_log_softmax, dim=1, keepdim=True)  # (num of cur, 1)
            
            if entropy < entropy_thresh * math.log2(histoty_feats.shape[0]):
                return True
       
        return False
    
    def _check_whether_push_2(self, cos_sim: Tensor, entropy: Tensor, 
                              cur_idxes: Tensor, hist_idxes: Tensor, 
                              rule: str = 'entropy',  # 'sim' or 'entropy',
                              sim_thresh: list = [0.2, 0.9], entropy_thresh: float = 0.45
                              ) -> bool:
        
        cos_sim_ = cos_sim[:, hist_idxes]
        cos_sim_ = cos_sim_[cur_idxes, :]

        entropy_ = entropy[:, hist_idxes]
        entropy_ = entropy_[cur_idxes, :]

        if rule == 'sim':
            if cos_sim_.min() > sim_thresh[0] and cos_sim_.max() < sim_thresh[1]:
                return True

        elif rule == 'entropy':
            if entropy_.mean() >= entropy_thresh:
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
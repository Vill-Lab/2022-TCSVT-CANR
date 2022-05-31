import numpy as np
import torch

from ..builder import BBOX_SAMPLERS
from .random_sampler import RandomSampler
import math


@BBOX_SAMPLERS.register_module()
class BalancedPosSamplerWithGT(RandomSampler):
    """Instance balanced sampler that samples equal number of positive samples
    for each instance."""

    def __init__(self, num, pos_fraction, neg_pos_ub=-1, add_gt_as_proposals=True, **kwargs):
        exceed_strategy = kwargs.pop('exceed_strategy', "add_all")
        assert exceed_strategy in ["add_all", "sample"]
        super().__init__(num, pos_fraction, neg_pos_ub, add_gt_as_proposals, **kwargs)
        self.exceed_strategy = exceed_strategy

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Sample positive boxes.

        Args:
            assign_result (:obj:`AssignResult`): The assigned results of boxes.
            num_expected (int): The number of expected positive samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        gt_flags = kwargs.get('gt_flags', None)
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            unique_gt_inds = assign_result.gt_inds[pos_inds].unique()
            num_gts = len(unique_gt_inds)
            num_per_gt = int(round(num_expected / float(num_gts)))
            if num_per_gt == 0:
                num_per_gt = 1
                if self.exceed_strategy == "sample":
                    unique_gt_inds = self.random_choice(unique_gt_inds, num_expected)
            sampled_inds_ = []
            for i in unique_gt_inds:
                inds = torch.nonzero(
                    assign_result.gt_inds == i.item(), as_tuple=False)
                gt_idx = None
                if inds.numel() != 0:
                    inds = inds.squeeze(1)
                    gt_flag = gt_flags[inds]
                    gt_idx = torch.nonzero(gt_flag, as_tuple=False)
                    if gt_idx.numel() != 0:
                        gt_idx = gt_idx.item()
                        gt_idx = inds[gt_idx]
                    else:
                        import ipdb; ipdb.set_trace()
                else:
                    import ipdb; ipdb.set_trace()
                    continue
                if gt_idx is None:
                    import ipdb; ipdb.set_trace()
                if len(inds) > num_per_gt:
                    inds = self.random_choice(inds, num_per_gt)
                    if torch.all(inds != gt_idx):
                        inds[0] = gt_idx
                sampled_inds_.append(inds)
            sampled_inds = torch.cat(sampled_inds_)
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(
                    list(set(pos_inds.cpu().numpy().tolist()) - set(sampled_inds.cpu().numpy().tolist())))
                if len(extra_inds) > num_extra:
                    extra_inds = self.random_choice(extra_inds, num_extra)
                extra_inds = torch.from_numpy(extra_inds).to(
                    assign_result.gt_inds.device).long()
                sampled_inds = torch.cat([sampled_inds, extra_inds])
            elif len(sampled_inds) > num_expected:
                pass
#                sampled_inds = self.random_choice(sampled_inds, num_expected)
            assert sampled_inds.size(0) == sampled_inds.unique().size(0)
            if sampled_inds[gt_flags[sampled_inds] == 1].size(0) != torch.sum(gt_flags).item() and self.exceed_strategy=="add_all":
                import ipdb; ipdb.set_trace()
            return sampled_inds

import math
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import get_dist_info
from torch.autograd import Function


def gather_tensors(inp):
    rank, world_size = get_dist_info()
    if world_size == 1:
        return inp
    local_size = torch.tensor([inp.size(0)], device=inp.device)
    input_shape = list(inp.shape)
    size_list = [torch.tensor([0], device=inp.device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(each.item()) for each in size_list]

    max_size = max(size_list)
    tensor_size = [max_size] + input_shape[1:]
    tensor_list = [torch.empty(tensor_size, dtype=inp.dtype, device=inp.device) for _ in range(world_size)]
    if local_size.item() < max_size:
        padded = torch.empty(tensor_size, dtype=inp.dtype, device=inp.device)
        padded[:input_shape[0], ...] = inp
    else:
        padded = inp
    dist.all_gather(tensor_list, padded)
    for i, each in enumerate(size_list):
        if each < max_size:
            tensor_list[i] = tensor_list[i][:each, ...]
    tensor_list = torch.cat(tensor_list, dim=0)
    return tensor_list


class OIMLoss(torch.nn.Module):
    """
    cls_id for labeled identity [0, num_pid - 1]
    cls_id for unlabeled identity `num_pid`
    cls_id for background `num_pid + 1`
    """

    class MatchFunction(Function):
        @staticmethod
        def forward(ctx, *inputs):
            inputs, targets, gt_flag, lut, queue, momentum, gt_only = inputs
            gt_only = torch.tensor(gt_only)
            if lut.dim() == 3:
                lut_ = torch.sum(lut, dim=1)
                lut_ = torch.nn.functional.normalize(lut_, dim=1)
            else:
                lut_ = lut
            ctx.save_for_backward(inputs, targets, gt_flag, lut, lut_, queue, momentum, gt_only)
            outputs_labeled = inputs.mm(lut_.t())
            outputs_unlabeled = inputs.mm(queue.t())
            return torch.cat((outputs_labeled, outputs_unlabeled), 1)

        @staticmethod
        def backward(ctx, *grad_outputs):
            grad_outputs, = grad_outputs
            inputs, targets, gt_flag, lut, lut_, queue, momentum, gt_only = ctx.saved_tensors
            gt_only = gt_only.item()
            inputs = gather_tensors(inputs)
            targets = gather_tensors(targets)
            gt_flag = gather_tensors(gt_flag)
            momentum = momentum.item()
            num_pid = lut.shape[0]
            grad_inputs = grad_outputs.mm(torch.cat((lut_, queue), 0))
            neg = inputs[targets.view(-1) < 0]
            if queue.size(0) > 0:
                queue[...] = torch.cat((queue[neg.shape[0]:], neg), 0)
            if lut.dim() == 2:
                for i, (x, y) in enumerate(zip(inputs, targets)):
                    if -1 < y < num_pid:
                        if gt_only and not gt_flag[i]:
                            continue
                        lut[y] = momentum * lut[y] + (1. - momentum) * x
                        lut[y] /= lut[y].norm()
            else:
                for i, (x, y) in enumerate(zip(inputs, targets)):
                    if -1 < y < num_pid:
                        if gt_only and not gt_flag[i]:
                            continue
                        lut[y, ...] = torch.cat((x.view(1, -1), lut[y, :-1, :]))

            return grad_inputs, None, None, None, None, None, None

    def __init__(self, num_pid, size_queue, reid_dim=256, momentum=0.5, temperature=10, adaptive_scale=False,
                 scale_multi=None, arcface=None, ce="normal", ce_threshold=0.75, gt_update_only=False, lut_inst=1, weight_mat=None, rand=True, rand_th=None, smoothing=0.1):
        super().__init__()
        self.reid_dim = reid_dim
        self.momentum = torch.scalar_tensor(momentum)
        self.num_pid = num_pid
        self.queue_size = size_queue
        self.lut_inst = lut_inst
        self.adaptive_scale = adaptive_scale
        self.scale_multi = scale_multi
        self.ce_threshold = ce_threshold
        if self.lut_inst > 1:
            self.register_buffer("lut", torch.zeros(self.num_pid, self.lut_inst, self.reid_dim))
        else:
            self.register_buffer("lut", torch.zeros(self.num_pid, self.reid_dim))
        self.register_buffer("queue", torch.zeros(self.queue_size, self.reid_dim))
        self.oim = OIMLoss.MatchFunction.apply
        if not self.adaptive_scale:
            self.register_buffer("temperature", torch.tensor(temperature, dtype=torch.float))
        else:
            self.register_buffer("temperature", torch.tensor([1.414 * math.log(num_pid + size_queue)]))
        self.update_gt_only = gt_update_only
        self.arcface = ArcSoftmax(**arcface) if arcface is not None else None
        if ce == "weighted":
            assert weight_mat is not None
            self.ce = WeightedCrossEntropy(torch.load(weight_mat).cuda(), num_pid=num_pid, threshold=self.ce_threshold, rand=rand, rand_th=rand_th)
        elif ce == "weighted2":
            assert weight_mat is not None
            self.ce = WeightedCrossEntropy2(torch.load(weight_mat).cuda(), num_pid=num_pid, threshold=self.ce_threshold, rand=rand, rand_th=rand_th)
        elif ce == "weighted3":
            assert weight_mat is not None
            self.ce = NRCE(torch.load(weight_mat).cuda(), num_pid=num_pid, threshold=self.ce_threshold)
        elif ce == "ls":
            self.ce = LabelSmoothing(smoothing=smoothing)
        else:
            self.ce = F.cross_entropy

    def forward(self, reid_feat, target, flags_):
        gt_flag = flags_ == 1
        reid_score = self.oim(reid_feat, target, gt_flag, self.lut, self.queue, self.momentum, self.update_gt_only)
        mask = (target >= 0) & (target <= self.num_pid)
        if torch.sum(mask) == 0:
            return torch.tensor(0.0, device=reid_feat.device)
        target = target[mask]
        reid_score = reid_score[mask]
        new_scale = None
        if self.adaptive_scale and target.size(0) > 0:
            # import ipdb; ipdb.set_trace()
            reid_scores = gather_tensors(reid_score)
            targets = gather_tensors(target)
            b = torch.sum(torch.exp(reid_scores * self.temperature.item()), dim=1) - torch.exp(
                reid_scores[torch.arange(0, targets.size(0)), targets] * self.temperature.item())
            cos_t = reid_scores[torch.arange(0, targets.size(0)), targets]
            len_t = cos_t.size(0)
            med_cos_t = torch.cos(0.5 * torch.arccos(cos_t[len_t // 2]) + 0.5 * torch.arccos(cos_t[(len_t - 1) // 2]))
            med = torch.max(torch.tensor(0.707, dtype=cos_t.dtype, device=cos_t.device), med_cos_t)
            new_scale = torch.log(torch.mean(b)) / med
            new_scale = new_scale.detach()

        if self.arcface:
            logits = self.arcface(reid_score, target)
        else:
            logits = reid_score * self.temperature.item()
        if self.scale_multi is not None:
            logits *= self.scale_multi
        if new_scale is not None:
            self.temperature[0] = new_scale
        reid_loss = self.ce(logits, target.view(-1))
        return reid_loss


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()



class noise_supressor(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, row, col, values=-100):
        ctx.save_for_backward(row, col)
        x[row, col] = -100
        return x

    @staticmethod
    def backward(ctx, grad_output):
        row, col = ctx.saved_tensors
        grad_output[row, col] *= 0.0
        return grad_output, None, None, None


class NRCE(nn.Module):

    def __init__(self, lut, num_pid=5532, threshold=0.75):
        super(NRCE, self).__init__()
        # import ipdb; ipdb.set_trace()
        self.num_pid = num_pid
        sim_mat = lut @ lut.T
        sim_mat[torch.arange(0, num_pid), torch.arange(0, num_pid)] = 0
        self.max_val = torch.max(sim_mat, dim=1).values
        self.max_ind = torch.max(sim_mat, dim=1).indices
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.threshold = threshold
        print("USING NRCE")

    def forward(self, logits, label):
        mask_labeled = label < self.num_pid
        clamped_label = torch.clamp_max(label, self.num_pid - 1)
        val = self.max_val[clamped_label]
        ind = self.max_ind[clamped_label]
        sim_ignore = (val > self.threshold)
        ignore = sim_ignore & mask_labeled

        if torch.sum(ignore) > 0:
            row = torch.where(ignore)[0]
            col = ind[ignore]
            logits = noise_supressor.apply(logits, row, col)
        loss = torch.nn.functional.cross_entropy(logits, target=label, ignore_index=self.num_pid)
        return loss



class WeightedCrossEntropy2(nn.Module):

    def __init__(self, lut, num_pid=5532, threshold=0.75, rand=True, rand_th=None):
        super(WeightedCrossEntropy2, self).__init__()
        # import ipdb; ipdb.set_trace()
        self.num_pid = num_pid
        sim_mat = lut @ lut.T
        sim_mat[torch.arange(0, num_pid), torch.arange(0, num_pid)] = 0
        self.max_val = torch.max(sim_mat, dim=1).values
        self.max_ind = torch.max(sim_mat, dim=1).indices
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.threshold = threshold
        self.rand = rand
        self.rand_th = threshold if rand_th is None else rand_th
        print("USING WeightedCrossEntropyV2")

    def forward(self, logits, label):
        mask_labeled = label < self.num_pid
        clamped_label = torch.clamp_max(label, self.num_pid - 1)
        val = self.max_val[clamped_label]
        ind = self.max_ind[clamped_label]
        if self.rand:
            sim_ignore = (val > random.random()) & (val > self.rand_th)
        else:
            sim_ignore = (val > self.threshold)
        ignore = sim_ignore & mask_labeled

        if torch.sum(ignore) > 0:
            row = torch.where(ignore)[0]
            col = ind[ignore]
            logits[row, col] *= 0.0
        log_logits = self.log_softmax(logits)
        loss = torch.nn.functional.nll_loss(log_logits, target=label, ignore_index=self.num_pid)
        return loss


class WeightedCrossEntropy(nn.Module):
    def __init__(self, lut, num_pid=5532, threshold=0.75, rand=True, rand_th=None):
        super(WeightedCrossEntropy, self).__init__()
        # import ipdb; ipdb.set_trace()
        self.num_pid = num_pid
        sim_mat = lut @ lut.T
        sim_mat[torch.arange(0, num_pid), torch.arange(0, num_pid)] = 0
        self.max_val = torch.max(sim_mat, dim=1).values
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.threshold = threshold
        self.rand = rand
        print("USING WeightedCrossEntropy")

    def forward(self, logits, label):
        # import ipdb; ipdb.set_trace()
        log_logits = self.log_softmax(logits)
        loss = torch.nn.functional.nll_loss(log_logits, target=label, ignore_index=self.num_pid, reduction="none")
        mask_labeled = label < self.num_pid
        clamped_label = torch.clamp_max(label, self.num_pid - 1)
        val = self.max_val[clamped_label]
        if self.rand:
            sim_ignore = (val > random.random()) & (val > self.threshold)
        else:
            sim_ignore = (val > self.threshold)
        valid = (~sim_ignore) & mask_labeled

        loss[~valid] *= 0
        n_valid = torch.sum(valid)
        if n_valid > 0:
            loss = loss.sum() / n_valid
        else:
            loss = loss.sum()

        return loss


class ArcSoftmax(nn.Module):
    def __init__(self, scale, margin):
        super().__init__()
        self.s = scale
        self.m = margin

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.threshold = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.register_buffer('t', torch.zeros(1))

    def forward(self, cos_theta, targets):
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        target_logit = cos_theta[torch.arange(0, cos_theta.size(0)), targets].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)
        pred_class_logits = cos_theta * self.s
        return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.s, self.m
        )


if __name__ == '__main__':
    feat = torch.randn((64, 256))
    oim_loss = OIMLoss(16, 16)
    target = torch.randint(-1, 16, (64,))
    feat.requires_grad_(True)
    print(feat)
    print(target)
    loss = oim_loss(feat, target)
    loss.backward()
    print(oim_loss.lut)
    print(oim_loss.queue)
    print(feat)

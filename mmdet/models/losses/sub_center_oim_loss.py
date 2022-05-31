import torch
import torch.nn as nn
from mmcv.runner import get_dist_info
from torch.autograd import Function
import torch.nn.functional as F
import torch.distributed as dist
import math

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


class SubcenterOIMLoss(torch.nn.Module):
    """
    cls_id for labeled identity [0, num_pid - 1]
    cls_id for unlabeled identity `num_pid`
    cls_id for background `num_pid + 1`
    """

    class MatchFunction(Function):
        @staticmethod
        def forward(ctx, *inputs):
            # import ipdb; ipdb.set_trace()
            inputs, targets, lut, queue, momentum, num_subcenters = inputs
            outputs_labeled = inputs.mm(lut.t())
            bz = outputs_labeled.size(0)
            pos_subcenters = torch.max(outputs_labeled.view(bz, -1, num_subcenters), dim=-1).indices
            ctx.save_for_backward(inputs, targets, lut, queue, momentum, torch.scalar_tensor(num_subcenters), pos_subcenters)
            return inputs

        @staticmethod
        def backward(ctx, *grad_outputs):
            # import ipdb; ipdb.set_trace()
            grad_input, = grad_outputs  # 512 * 10532
            inputs, targets, lut, queue, momentum, num_subcenters, pos_subcenters = ctx.saved_tensors
            num_subcenters = int(num_subcenters.item())
            num_pid = lut.shape[0] // num_subcenters
            pos_subcenter = pos_subcenters[torch.arange(0, pos_subcenters.size(0)),targets.clamp(max=num_pid - 1)]  # 512 * 5532
            inputs = gather_tensors(inputs)
            targets = gather_tensors(targets)
            all_pos_subcenter = gather_tensors(pos_subcenter)
            momentum = momentum.item()

            neg = inputs[targets.view(-1) == num_pid]
            queue[...] = torch.cat((queue[neg.shape[0]:], neg), 0)
            for i, (x, y) in enumerate(zip(inputs, targets)):
                if -1 < y < num_pid:
                    idx = y * num_subcenters + all_pos_subcenter[i]
                    lut[idx] = momentum * lut[idx] + (1. - momentum) * x
                    lut[idx] /= lut[idx].norm()
            return grad_input, None, None, None, None, None, None

    def __init__(self, num_pid, size_queue, reid_dim=256, momentum=0.5, temperature=10, arcface=None, subcenters=2):
        super().__init__()
        self.reid_dim = reid_dim
        self.momentum = torch.scalar_tensor(momentum)
        self.subcenters = subcenters
        self.num_pid = num_pid
        self.queue_size = size_queue
        self.register_buffer("lut", torch.zeros(self.num_pid * self.subcenters, self.reid_dim))
        self.register_buffer("queue", torch.zeros(self.queue_size, self.reid_dim))
        nn.init.xavier_uniform_(self.lut)
        self.lut /= self.lut.norm(dim=1, keepdim=True)
        # nn.init.xavier_uniform_(self.queue)
        self.oim = SubcenterOIMLoss.MatchFunction.apply
        self.temperature = temperature
        self.arcface = ArcSoftmax(**arcface) if arcface is not None else None



    def forward(self, reid_feat, target):
        reid_feat = self.oim(reid_feat, target, self.lut, self.queue, self.momentum, self.subcenters)
        outputs_labeled = reid_feat.mm(self.lut.t())
        bz, dim = outputs_labeled.size()
        outputs_labeled = torch.nn.functional.max_pool1d(outputs_labeled.view(bz, 1, dim), self.subcenters).squeeze(1)
        outputs_unlabeled = reid_feat.mm(self.queue.t())
        reid_score = torch.cat((outputs_labeled, outputs_unlabeled), 1)
        
        mask = (target >= 0) & (target < self.num_pid)
        target = target[mask]
        reid_score = reid_score[mask]
        if self.arcface:
            logits = self.arcface(reid_score, target)
        else:
            logits = reid_score * self.temperature
        reid_loss = F.cross_entropy(logits, target.view(-1))
        return reid_loss


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

import torch
import torch.distributed as dist
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


class OICLoss(torch.nn.Module):
    """
    compare unlbl person with labeled person

    cls_id for labeled identity [0, num_pid - 1]
    cls_id for unlabeled identity `num_pid`
    cls_id for background `num_pid + 1`
    """

    class MatchFunction(Function):

        @staticmethod
        def forward(ctx, *inputs):
            inputs, targets, queue, num_pid = inputs
            ctx.save_for_backward(inputs, targets, queue, torch.scalar_tensor(num_pid, dtype=torch.long))
            output = inputs.mm(queue.t())
            return output

        @staticmethod
        def backward(ctx, *grad_outputs):
            grad_outputs, = grad_outputs
            inputs, targets, queue, num_pid = ctx.saved_tensors
            num_pid = num_pid.item()
            inputs = gather_tensors(inputs)
            targets = gather_tensors(targets)
            grad_inputs = grad_outputs.mm(queue)
            pos_mask = (targets.view(-1) >= 0) & (targets.view(-1) < num_pid)
            pos = inputs[pos_mask]
            queue[...] = torch.cat((queue[pos.shape[0]:], pos), 0)
            return grad_inputs, None, None, None, None, None

    def __init__(self, size_queue, reid_dim=256, temperature=10, num_pid=5532):
        super(OICLoss, self).__init__()
        self.reid_dim = reid_dim
        self.num_pid = num_pid
        self.queue_size = size_queue
        self.register_buffer("queue", torch.zeros(self.queue_size, self.reid_dim))
        self.xoim = OICLoss.MatchFunction.apply
        self.temperature = temperature

    def forward(self, reid_feat, target):
        cmp_score = self.xoim(reid_feat, target, self.queue, self.num_pid)
        valid_mask = target < 0
        if torch.sum(valid_mask) == 0:
            return torch.tensor(0.0, device=reid_feat.device)
        cmp_score = torch.cat([torch.ones(torch.sum(valid_mask).item(), 1, dtype= cmp_score.dtype, device=cmp_score.device), cmp_score[valid_mask]], dim=1)
        logits = cmp_score * self.temperature
        target = torch.zeros(torch.sum(valid_mask), dtype=torch.long, device=logits.device)
        reid_loss = F.cross_entropy(logits, target)
        return reid_loss


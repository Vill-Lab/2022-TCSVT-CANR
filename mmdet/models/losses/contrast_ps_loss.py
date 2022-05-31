import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.runner import get_dist_info
from torch.nn import ModuleDict

from mmdet.models import build_loss
from .oim_loss import gather_tensors


class grad_scale_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return lambda_ * grad_input, None


class GradScale(nn.Module):
    def __init__(self, lambda_=1.):
        super(GradScale, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return grad_scale_func.apply(x, self.lambda_)


class ContrastivePersonSearchLoss(nn.Module):

    def _calc(self, num_cls):
        return 1.414 * math.log(num_cls)

    def __init__(self, lambda_, scale_a=2.5, scale_r=2.5, scale_ax=5, scale_rx=5, loss_policy=3, dbg=False, loss_cfg=None, logit_bn=False, norm_type="BN1d",
                 *args, **kwargs):
        # set scale to none to use adaptive scale
        super(ContrastivePersonSearchLoss, self).__init__()
        self.grad_scale = GradScale(lambda_)
        self.scale_a = scale_a
        self.scale_r = scale_r
        self.scale_ax = scale_ax
        self.scale_rx = scale_rx
        self.loss_policy = loss_policy
        self.dbg = dbg
        self.loss_override = ModuleDict()
        self.logit_bn = ModuleDict()
        if loss_cfg:
            for k, v in loss_cfg:
                self.loss_override[k] = build_loss(v)
                if logit_bn:
                    self.logit_bn[k] = build_norm_layer({"type": norm_type, "eps": 2e-5}, 1)[1]


    def contrastive_ps_loss_single(self, embeddings, label, gt_flag, fgbg_flag):
        # import ipdb; ipdb.set_trace()
        embedding_fg = embeddings[fgbg_flag]
        label = label[fgbg_flag]
        gt_flag = gt_flag[fgbg_flag]
        embedding_gt_ = embedding_fg[gt_flag]
        embedding_gt = self.grad_scale(embedding_gt_)
        embedding_proposal = embedding_fg[~gt_flag]
        # reassign_label
        label_gt = label[gt_flag]
        label_proposal = label[~gt_flag]
        label_mapper = dict()
        for l, o in enumerate(label_gt.cpu().numpy().tolist()):
            label_mapper[int(o)] = l
        mapped_label = label_proposal.clone()
        for i in range(mapped_label.size(0)):
            true_label = label_mapper.get(int(mapped_label[i].item()), None)
            if true_label is None:
                import ipdb;
                ipdb.set_trace()
            mapped_label[i] = true_label
        if self.scale_a > 0:
            logits_comp = (embedding_proposal @ embedding_gt.T) * self.scale_a
        else:
            logits_comp = (embedding_proposal @ embedding_gt.T) * self.scale_a * self._calc(
                torch.sum(gt_flag).item()) * -1
        if "0x01" in self.loss_override.keys():
            if "0x01" in self.logit_bn.keys():
                sz = logits_comp.size()
                logits_comp = self.logit_bn["0x01"](logits_comp.view(-1, 1)).view(sz)
            proposal_contrastive_loss = self.loss_override["0x01"](logits_comp, mapped_label)
        else:
            proposal_contrastive_loss = F.cross_entropy(logits_comp, mapped_label)

        if self.scale_r > 0:
            logits_gt = (embedding_gt_ @ embedding_gt_.T) * self.scale_r
        else:
            logits_gt = (embedding_gt_ @ embedding_gt_.T) * self.scale_r * self._calc(torch.sum(gt_flag).item()) * -1
        targets_gt = torch.arange(0, label_gt.size(0), dtype=label_gt.dtype, device=label_gt.device)
        if "0x02" in self.loss_override.keys():
            if "0x02" in self.logit_bn.keys():
                sz = logits_gt.size()
                logits_gt = self.logit_bn["0x02"](logits_gt.view(-1, 1)).view(sz)
            gt_contrastive_loss = self.loss_override["0x02"](logits_gt, targets_gt)
        else:
            gt_contrastive_loss = F.cross_entropy(logits_gt, targets_gt)
        return proposal_contrastive_loss, gt_contrastive_loss

    def batch_prop_cont_loss(self, embedding, label, gt_flags, fgbg_flag, source):
        rank, world_size = get_dist_info()
        if world_size > 1:
            single_batch_size = embedding.size(0)
            batch_sizes = [torch.tensor([0], dtype=torch.long, device=embedding.device) for _ in range(world_size)]
            dist.all_gather(batch_sizes, torch.tensor([single_batch_size], device=embedding.device))
            batch_sizes = torch.tensor(batch_sizes, device=embedding.device)
            gathered_embedding = gather_tensors(embedding)
            embedding = torch.cat([gathered_embedding[:torch.sum(batch_sizes[:rank]).item(), :],
                                   embedding,
                                   gathered_embedding[torch.sum(batch_sizes[:rank + 1]).item():, :]])
            label = gather_tensors(label)
            gt_flags = gather_tensors(gt_flags)
            fgbg_flag = gather_tensors(fgbg_flag)
            rank, world_size = get_dist_info()
            source += rank * 16
            source = gather_tensors(source)
        lbl_gt_flags = gt_flags & (label >= 0)
        ulbl_gt_flags = gt_flags & (label < 0)
        lbl_gt = label[lbl_gt_flags]
        # import ipdb; ipdb.set_trace()
        lbl_ids = torch.unique(lbl_gt)
        lbl_embedding = []
        lbl_target = []
        for e in lbl_ids:
            if torch.sum(lbl_gt == e) > 1:
                emb = embedding[(label == e) & gt_flags]
                emb = torch.nn.functional.normalize(torch.sum(emb, dim=0, keepdim=True), dim=1)
                lbl_embedding.append(emb)

            else:
                emb = embedding[(label == e) & gt_flags]
                lbl_embedding.append(emb)
            lbl_target.append(e.item())
        # import ipdb; ipdb.set_trace()
        all_embeddings = []
        if lbl_embedding:
            all_embeddings.append(torch.cat(lbl_embedding, dim=0))
        ulbl_embedding = embedding[ulbl_gt_flags]
        # import ipdb; ipdb.set_trace()
        if ulbl_embedding.size(0) > 0:
            all_embeddings.append(ulbl_embedding)
        # import ipdb; ipdb.set_trace()
        gt_embedding_ = torch.cat(all_embeddings, dim=0)

        losses = dict()
        if self.loss_policy & 0x04:
            ulbl_target = [(l.item(), s.item()) for l, s in zip(label[ulbl_gt_flags], source[ulbl_gt_flags])]
            target_mapper = dict()
            for i, target in enumerate(lbl_target + ulbl_target):
                target_mapper[target] = i

            proposal_flag = fgbg_flag & (~gt_flags)
            proposal_embedding = embedding[proposal_flag]
            proposal_target = []
            for l, s in zip(label[proposal_flag], source[proposal_flag]):
                l = l.item()
                s = s.item()
                if l >= 0:
                    proposal_target.append(target_mapper[l])
                else:
                    proposal_target.append(target_mapper[(l, s)])
            proposal_target = torch.as_tensor(proposal_target, dtype=label.dtype, device=label.device)
            gt_embedding = self.grad_scale(gt_embedding_)
            if self.scale_ax > 0:
                xp_logit = (proposal_embedding @ gt_embedding.T) * self.scale_ax
            else:
                xp_logit = (proposal_embedding @ gt_embedding.T) * self.scale_ax * self._calc(
                    torch.sum(gt_flags).item()) * -1
            if "0x04" in self.loss_override.keys():
                if "0x04" in self.logit_bn.keys():
                    sz = xp_logit.size()
                    xp_logit = self.logit_bn["0x04"](xp_logit.view(-1, 1)).view(sz)
                xp_loss = self.loss_override["0x04"](xp_logit, proposal_target)
            else:
                xp_loss = F.cross_entropy(xp_logit, proposal_target)
            if not torch.isnan(xp_loss):
                losses["xp_contrastive_loss"] = xp_loss
            elif self.dbg:
                import ipdb; ipdb.set_trace()
        if self.loss_policy & 0x08:
            gt_target = torch.arange(0, gt_embedding_.size(0), dtype=label.dtype, device=label.device)
            if self.scale_rx > 0:
                xgt_logits = (gt_embedding_ @ gt_embedding_.T) * self.scale_rx
            else:
                xgt_logits = (gt_embedding_ @ gt_embedding_.T) * self.scale_rx * self._calc(
                    torch.sum(gt_flags).item()) * -1
            if "0x08" in self.loss_override.keys():
                if "0x08" in self.logit_bn.keys():
                    sz = xgt_logits.size()
                    xgt_logits = self.logit_bn["0x08"](xgt_logits.view(-1, 1)).view(sz)
                xgt_loss = self.loss_override["0x08"](xgt_logits, gt_target)
            else:
                xgt_loss = F.cross_entropy(xgt_logits, gt_target)
            if not torch.isnan(xgt_loss):
                losses["xgt_contrastive_loss"] = xgt_loss
            elif self.dbg:
                import ipdb; ipdb.set_trace()

        return losses

    def forward(self, embeddings, label, flags_, fgbg_label, rois):
        # import ipdb; ipdb.set_trace()
        fgbg_flag = fgbg_label == 0
        source = rois[:, 0].type(torch.uint8)
        gt_flag = flags_ == 1
        im_per_batch = torch.max(source).item() + 1
        proposal_contrastive_loss = []
        gt_contrastive_loss = []
        losses = dict()
        if self.loss_policy & 0x03:
            for i in range(im_per_batch):
                # import ipdb; ipdb.set_trace()
                cur_im_mask = source == i
                prop_loss, gt_loss = self.contrastive_ps_loss_single(embeddings[cur_im_mask], label[cur_im_mask],
                                                                     gt_flag[cur_im_mask], fgbg_flag[cur_im_mask])
                proposal_contrastive_loss.append(prop_loss)
                gt_contrastive_loss.append(gt_loss)
            if self.loss_policy & 0x01:
                p_loss = sum(proposal_contrastive_loss) / len(proposal_contrastive_loss)
                if not torch.isnan(p_loss):
                    losses["p_contrastive_loss"] = p_loss
            if self.loss_policy & 0x02:
                gt_loss = sum(gt_contrastive_loss) / len(gt_contrastive_loss)
                if not torch.isnan(gt_loss):
                    losses["gt_contrastive_loss"] = gt_loss
        if self.loss_policy & 0x0c:
            losses.update(self.batch_prop_cont_loss(embeddings, label, gt_flag, fgbg_flag, source))
        return losses


if __name__ == '__main__':
    l = ContrastivePersonSearchLoss(0.1, loss_policy=1)
    embedding = F.normalize(torch.randn(16, 9))
    label = torch.arange(0, 16)
    flags = torch.tensor([1 for _ in range(16)])
    fgbg_flag = torch.tensor([0 for _ in range(16)])
    print(l(embedding, label, flags, fgbg_flag, torch.zeros((16, 1), dtype=torch.int32)))

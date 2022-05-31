import math

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2dPack as DCNv2

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet.models.roi_heads import StandardRoIHead
from ..builder import HEADS, build_head, build_roi_extractor, build_shared_head


def interpolate(to_interpolate, target_size, mode="bilinear", align_corners=False):
    return torch.nn.functional.interpolate(
        to_interpolate[None, None, ...], size=target_size, mode=mode,
        align_corners=align_corners).squeeze()


def calc_or_loss1(common_area1, common_area2, param_a, param_b):
    return torch.mean(
        torch.pow(common_area1 * common_area2, param_a) * torch.pow(common_area1 + common_area2, param_b))


def calc_or_loss2(common_area1, common_area2, param_a, param_b):
    return torch.mean(torch.relu(common_area1 + common_area2 - param_a) * param_b)


def calc_or_loss3(common_area1, common_area2, param_a, param_b):
    r = (2.0 - param_a) * math.sqrt(2) / 2 + param_b
    c_circle = 1 + math.sqrt(2) / 2 * param_b
    a = torch.norm(torch.stack([common_area1 - c_circle, common_area2 - c_circle], dim=0), dim=0)
    t = torch.where(a < r, a, torch.tensor(r, device=a.device))
    return torch.mean(r - torch.sqrt(r ** 2 - torch.pow(t - r, 2)))


def calc_or_loss4(common_area1, common_area2, param_a, param_b):
    top_c = 2
    output = torch.stack([common_area1, common_area2], dim=-1).view(-1, 2)
    bat = output.shape[0]
    hloss = F.softmax(output, dim=1) * F.log_softmax(output, dim=1)
    hloss = torch.mean(hloss, 1)

    hloss = (-1.0 * hloss.sum()) * output.shape[1] / (
            torch.log(torch.Tensor([output.shape[1]]).cuda()) * output.shape[0])

    output, _ = output.topk(top_c, 1, True, True)
    # print(output.shape)
    m = D.multivariate_normal.MultivariateNormal(torch.zeros(output.shape[1]).cuda(),
                                                 hloss * torch.eye(output.shape[1]).cuda())
    # x = F.softmax(x, dim=1)
    # output = F.softmax(output, dim=1)
    loss = m.log_prob(output).exp().sum() / bat
    return loss  # 0 <= loss <= log(n)/n*BS


def calc_or_loss5(common_area1, common_area2, param_a, param_b):
    return torch.mean(
        torch.exp(common_area1 * common_area2 * param_a) - param_b * torch.pow(common_area1 - common_area2, 2)
    )


def calc_or_loss6(common_area1, common_area2, param_a, param_b):
    return torch.mean(torch.exp(param_a * common_area1 * common_area2) - param_b * torch.pow(common_area1 - 0.5,
                                                                                             2) - param_b * torch.pow(
        common_area2 - 0.5, 2))


class OcclusionRelationLoss(nn.Module):

    def __init__(self, a=0.5, b=0.5, loss_surface=1):
        super().__init__()
        self.iou_calculators = build_iou_calculator(dict(type='BboxOverlaps2D'))
        self.a = a
        self.b = b
        self.loss_surface = eval("calc_or_loss{}".format(loss_surface))

    def loss_single(self, mask1, mask2, box1, box2):
        mask1 = mask1.squeeze(0)
        mask2 = mask2.squeeze(0)
        mask_multi = 3
        x1, y1, x2, y2 = box1.cpu().numpy().tolist()
        a1, b1, a2, b2 = box2.cpu().numpy().tolist()
        bw1 = x2 - x1
        bh1 = y2 - y1
        bw2 = a2 - a1
        bh2 = b2 - b1
        tl_x, tl_y = max(x1, a1), max(y1, b1)
        br_x, br_y = min(x2, a2), min(y2, b2)
        mask_size = mask1.size(-1) * mask_multi
        mask1 = interpolate(mask1, (mask_size, mask_size))
        mask2 = interpolate(mask2, (mask_size, mask_size))

        st_x1 = math.floor((tl_x - x1) / bw1 * mask_size)
        ed_x1 = math.ceil((br_x - x1) / bw1 * mask_size)
        st_y1 = math.floor((tl_y - y1) / bh1 * mask_size)
        ed_y1 = math.ceil((br_y - y1) / bh1 * mask_size)

        st_x2 = math.floor((tl_x - a1) / bw2 * mask_size)
        ed_x2 = math.ceil((br_x - a1) / bw2 * mask_size)
        st_y2 = math.floor((tl_y - b1) / bh2 * mask_size)
        ed_y2 = math.ceil((br_y - b1) / bh2 * mask_size)

        common_area1 = mask1[st_y1:ed_y1, st_x1:ed_x1]
        common_area2 = mask2[st_y2:ed_y2, st_x2:ed_x2]

        if common_area1.size(0) == 0 or common_area1.size(1) == 0:
            return None

        if common_area2.size(0) == 0 or common_area2.size(1) == 0:
            return None

        comm_h = max(common_area1.size(0), common_area2.size(0))
        comm_w = max(common_area1.size(1), common_area2.size(1))
        common_area1 = interpolate(common_area1, (comm_h, comm_w))
        common_area2 = interpolate(common_area2, (comm_h, comm_w))

        loss = self.loss_surface(common_area1, common_area2, self.a, self.b)
        return loss

    def forward(self, mask, gt_bboxes, source):
        # import ipdb; ipdb.set_trace()
        source = source.int()
        iou = self.iou_calculators(gt_bboxes, gt_bboxes)
        unique = torch.unique(source)
        single_losses = []
        for e in unique:
            mask_bin = source == e
            cur_iou = iou[mask_bin][:, mask_bin]
            cur_mask = mask[mask_bin]
            cur_gt_bboxes = gt_bboxes[mask_bin]
            r = cur_iou.size(0)
            for i in range(r):
                for j in range(i + 1, r):
                    if cur_iou[i][j] > 1e-5:
                        single_loss = self.loss_single(cur_mask[i], cur_mask[j], cur_gt_bboxes[i], cur_gt_bboxes[j])
                        if single_loss is not None:
                            single_losses.append(single_loss)
        num = len(single_losses)
        if num > 0:
            loss = sum(single_losses) / float(num)
        else:
            loss = torch.mean(0.0 * mask[..., 0])
        return loss


class SpatialAttention(nn.Module):

    def margin_loss(self, sa, frac=0.3, margin=0.5, lam=0.99, reduce=None):
        # sa: tensor, (n, 1, h, w)
        sa = sa.flatten(start_dim=1)
        size = sa.size(1)
        k = int(size * frac)
        top = torch.topk(sa, k, dim=1).values
        bottom = torch.topk(sa, k, dim=1, largest=False).values

        ema = lam ** torch.arange(k - 1, -1, -1, dtype=sa.dtype, device=sa.device)
        ema = ema / torch.sum(ema)
        top = torch.sum(top * ema, dim=1)
        bottom = torch.sum(bottom * ema, dim=1)
        loss = torch.relu(bottom + margin - top)
        if reduce == "mean":
            loss = torch.mean(loss)
        elif reduce == "sum":
            loss = torch.sum(loss)
        return loss

    def consistency_loss(self, sa, reduce=None):
        sa = torch.sum(torch.abs(torch.nn.functional.conv2d(sa, self.consistency_conv)), dim=1)
        loss = torch.mean(sa, dim=(1, 2))
        if reduce == "mean":
            loss = torch.mean(loss)
        elif reduce == "sum":
            loss = torch.sum(loss)
        return loss

    def hard_attention_loss(self, sa, proposal, gt_bboxes, gt_flag):
        """
        proposal: [[source, x1, y1, x2, y2], ...]
        gt_bboxes: [[x1, y1, x2, y2], ...]
        gt_flag: [boolean, ...]
        """
        if torch.sum(gt_flag) > 0:
            proposal = proposal[gt_flag]
            gt_bboxes = gt_bboxes[gt_flag]
            sa_h, sa_w = sa.size(-2), sa.size(-1)
            sa = sa[gt_flag]
            n_sa = torch.sum(gt_flag).item()

            w = proposal[:, 3] - proposal[:, 1]
            h = proposal[:, 4] - proposal[:, 2]
            x1 = (torch.max(proposal[:, 1], gt_bboxes[:, 0]) - proposal[:, 1]) / w
            y1 = (torch.max(proposal[:, 2], gt_bboxes[:, 1]) - proposal[:, 2]) / h
            x2 = (torch.min(proposal[:, 3], gt_bboxes[:, 2]) - proposal[:, 1]) / w
            y2 = (torch.min(proposal[:, 4], gt_bboxes[:, 3]) - proposal[:, 2]) / h
            mask_h = sa_h * self.ha_emu
            mask_w = sa_w * self.ha_emu
            valid_mask = torch.zeros(n_sa, 1, mask_h, mask_w, device=sa.device, dtype=sa.dtype)
            for i in range(n_sa):
                cur_x1 = (x1[i] * mask_w).floor().long().item()
                cur_y1 = (y1[i] * mask_h).floor().long().item()
                cur_x2 = (x2[i] * mask_w).ceil().long().item()
                cur_y2 = (y2[i] * mask_h).ceil().long().item()
                width = cur_x2 - cur_x1
                height = cur_y2 - cur_y1
                if not (width > 0 and height > 0):
                    continue
                valid_mask[i, 0, cur_y1:cur_y2, cur_x1:cur_x2] = 1.0

            valid_mask = torch.nn.functional.interpolate(
                valid_mask, size=(sa_h, sa_w), mode='bilinear', align_corners=False)
            if self.dbg:
                if torch.any(torch.isnan(valid_mask)):
                    import ipdb;
                    ipdb.set_trace()
            else:
                assert not torch.any(torch.isnan(valid_mask))
            loss = torch.nn.functional.binary_cross_entropy(sa, valid_mask)
            return loss
        else:
            return None

    def __init__(self, kernel_size=7, ha_emu=3, dbg=False, extra_conv=False, in_channels=None):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 5, 7), 'kernel size must be 3 or 5 or 7'
        padding = kernel_size // 2
        assert (not extra_conv) or (in_channels is not None)
        self.extra_conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1,
                                    bias=False) if extra_conv else None
        self.extra_bn = nn.BatchNorm2d(1) if extra_conv else None

        self.conv1 = nn.Conv2d(2 + extra_conv, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.ha_emu = ha_emu
        self.dbg = dbg

        self.register_buffer("consistency_conv", torch.tensor([
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]]
        ], dtype=torch.float).view(4, 1, 3, 3))

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att_conv_in = [avg_out, max_out]
        if self.extra_conv is not None:
            att_conv_in.append(torch.relu(self.extra_bn(self.extra_conv(x))))

        x = torch.cat(att_conv_in, dim=1)
        x = self.conv1(x)
        result = self.sigmoid(x)
        if self.dbg:
            if torch.any(torch.isnan(result)):
                import ipdb;
                ipdb.set_trace()
        else:
            assert not torch.any(torch.isnan(result))
        return result


class CaC(nn.Module):

    def __init__(self, in_channels, s=3):
        super(CaC, self).__init__()
        self.in_channels = in_channels
        self.s = s
        self.tk = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.tq = nn.Conv2d(in_channels=in_channels, out_channels=s * s, kernel_size=1)
        self.bn = nn.BatchNorm2d(num_features=in_channels)

    def forward(self, x):
        n, c, h, w = x.size()
        feat_k = self.tk(x).view(n, c, h * w)
        feat_q = self.tq(x).view(n, self.s ** 2, h * w).transpose(1, 2)
        krnl = torch.bmm(feat_k, feat_q).view(n, c, self.s, self.s)
        krnl = self.bn(krnl).view(n * c, 1, self.s, self.s)
        x_ = x.view(1, n * c, h, w)
        a1 = torch.sigmoid(torch.nn.functional.conv2d(input=x_, weight=krnl, padding=1, dilation=1, groups=n * c))
        a2 = torch.sigmoid(torch.nn.functional.conv2d(input=x_, weight=krnl, padding=2, dilation=2, groups=n * c))
        a3 = torch.sigmoid(torch.nn.functional.conv2d(input=x_, weight=krnl, padding=3, dilation=3, groups=n * c))
        a = (a1 + a2 + a3) / 3.0
        a = a.view(n, c, h, w)
        return a


class FeatureFusionModule(nn.Module):
    def __init__(self, groups=32, policy=0x00):
        super(FeatureFusionModule, self).__init__()
        self.up = nn.ConvTranspose2d(2048, 2048, 4, 2, 1, groups=512) if policy & 0x01 else None
        self.align = DCNv2(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, groups=groups)
        mli = []
        if policy & 0x02:
            mli.append(torch.nn.BatchNorm2d(3072))
        if policy & 0x04:
            mli.append(nn.ReLU())
        mli.append(nn.Conv2d(in_channels=3072, out_channels=2048, kernel_size=1, stride=1, groups=256))
        if policy & 0x08:
            mli.append(torch.nn.BatchNorm2d(2048))
        if policy & 0x10:
            mli.append(nn.ReLU())
        self.post = nn.Sequential(*mli)

    def forward(self, lo_res, hi_res):
        if self.up is not None:
            lo_res_up = self.up(lo_res)
        else:
            lo_res_up = F.interpolate(lo_res, scale_factor=2.0, mode="bilinear", align_corners=False)
        hi_res_align = self.align(hi_res)
        x = torch.cat([lo_res_up, hi_res_align], dim=1)
        x = self.post(x)
        return x


class ContextAwareAttention(nn.Module):

    def __init__(self, in_channel, scale_ratio=8, detach=True, residual=True, pool_size=14):
        super(ContextAwareAttention, self).__init__()
        self.scene_conv = nn.Conv2d(in_channel, in_channel // scale_ratio, 1)
        self.proposal_conv = nn.Conv2d(in_channel, in_channel // scale_ratio, 1)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.in_channel = in_channel
        self.scale_ratio = scale_ratio
        self.pool_size = pool_size
        self.fc = nn.Linear(pool_size * pool_size, 1)
        self.detach = detach
        self.residual = residual

    def forward(self, proposal_ftmp, item_idx, scene_ftmp):
        item_idx = item_idx.int()
        if self.detach:
            scene_ftmp = scene_ftmp.detach()
        if self.pool is not None:
            scene_ftmp = self.pool(scene_ftmp)
        scene_ftmp = self.scene_conv(scene_ftmp)
        proposal_ftmp_ = self.proposal_conv(proposal_ftmp)
        attention_tensor = torch.zeros_like(proposal_ftmp)
        for i in item_idx.unique():
            cur_proposal = proposal_ftmp_[item_idx == i, ...]  # b' * c * h * w
            ib, ic, ih, iw = cur_proposal.shape
            r_item = cur_proposal.view(ib, ic, ih * iw).permute(0, 2, 1).reshape(ib * ih * iw, ic)
            cur_scene = scene_ftmp[i].reshape(self.in_channel // self.scale_ratio, -1)
            att = r_item @ cur_scene  # (ib * ih * iw) * (sh * sw)
            att = att.reshape(ib * ih * iw, self.pool_size * self.pool_size)
            # att = torch.nn.functional.softmax(att, dim=1).reshape(ib, ih, iw, self.pool_size * self.pool_size)
            att = torch.sigmoid(self.fc(att).reshape(ib, ih, iw))
            attention_tensor[item_idx == i, :, :, :] = att[:, None, :, :]

        result = attention_tensor * proposal_ftmp

        if self.residual:
            result += proposal_ftmp

        return result


@HEADS.register_module()
class PersonSearchRoIHead(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self, attention_policy=0, attention_loss_policy=0, attention_loss_weight=(1, 1, 1, 1, 1, 1),
                 attention_kernel=(7, 7), margin_att_loss_parameter=None, jitter_num=0, jitter_scale=0.05,
                 dual_head=False, dbg=False, use_ffm=False, ffm_group=32, ffm_policy=0x01, use_relation_loss=False,
                 rl_a=0.5, rl_b=0.5, relation_loss_policy=0x01, sa_extra_conv=False, extra_conv_policy=0x03,
                 loss_surface=1, caa=False, *args, **kwargs):
        super(PersonSearchRoIHead, self).__init__(*args, **kwargs)
        self.jitter_scale = jitter_scale
        self.jitter_num = jitter_num
        self.att_loss_policy = attention_loss_policy
        self.att_loss_weight = attention_loss_weight
        self.margin_att_loss_param = margin_att_loss_parameter if margin_att_loss_parameter is not None else [None,
                                                                                                              None]
        self.relation_loss = OcclusionRelationLoss(a=rl_a, b=rl_b,
                                                   loss_surface=loss_surface) if use_relation_loss else None
        self.relation_loss_policy = relation_loss_policy

        assert attention_policy in list(range(32))
        self.sa1 = SpatialAttention(attention_kernel[0], dbg=dbg,
                                    extra_conv=(sa_extra_conv and (extra_conv_policy & 0x01) > 0),
                                    in_channels=1024) if attention_policy & 0x02 else None
        self.sa2 = SpatialAttention(attention_kernel[1], dbg=dbg,
                                    extra_conv=(sa_extra_conv and (extra_conv_policy & 0x02) > 0),
                                    in_channels=2048) if attention_policy & 0x08 else None
        self.cac1 = CaC(in_channels=1024) if attention_policy & 0x01 else None
        self.cac2 = CaC(in_channels=2048) if attention_policy & 0x04 else None
        self.cac_ftmp = CaC(in_channels=1024) if attention_policy & 0x10 else None
        self.caa = ContextAwareAttention(1024) if caa else None
        self.det_head = build_shared_head(dict(
            type='ResLayers',
            depth=50,
            stage=(2, 3),
            stride=(2, 2),
            style='caffe',
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            offset=(5, 0),
            end_trim=(0, 2))) if dual_head else None
        self.ffm = FeatureFusionModule(groups=ffm_group, policy=ffm_policy) if use_ffm else None

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.det_head is not None:
            self.det_head.init_weights(pretrained=pretrained)
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'],)
        return outs

    def jitter_bbox(self, proposal_list, gt_bboxes, cnt, scale=0.05):
        jittered_gt = []
        for gt_bbox in gt_bboxes:
            num_bbox = gt_bbox.size(0)
            w, h = gt_bbox[:, 2] - gt_bbox[:, 0], gt_bbox[:, 3] - gt_bbox[:, 1]
            scaled = torch.stack([w, h, w, h]).T * scale
            jittered = torch.randn(num_bbox, cnt, 4, device=gt_bbox.device) * scaled[:, None, :] + gt_bbox[:, None,
                                                                                                   :]
            jittered_gt.append(jittered.reshape(-1, 4))

        proposal_list_with_jitter = []
        for j, p in zip(jittered_gt, proposal_list):
            j_with_score = torch.cat((j, torch.ones(j.size(0), 1, device=j.device)), dim=1)
            proposal_list_with_jitter.append(torch.cat((p, j_with_score), dim=0))
        return proposal_list_with_jitter

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.jitter_num != 0:
            proposal_list = self.jitter_bbox(proposal_list, gt_bboxes, self.jitter_num, self.jitter_scale)
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(  # xyxy
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.cac_ftmp:
            x = [self.cac_ftmp(x[0]) * x[0]]
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])
            losses.update(bbox_results['loss_attention'])
            if "info_loss" in bbox_results.keys() and bbox_results["info_loss"]:
                losses.update(loss_info=bbox_results["info_loss"])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            # TODO: Support empty tensor input. #2280
            if mask_results['loss_mask'] is not None:
                losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois, fgbg_flag=None):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        extracted_roi_ft = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        det_feat = self.det_head(extracted_roi_ft) if self.det_head else None
        assert self.with_shared_head
        sa_mask1 = None
        sa_mask2 = None
        if self.with_shared_head:
            # pre id-net attention
            ft_pre_shared_head = extracted_roi_ft
            if self.cac1:
                cac_mask1 = self.cac1(ft_pre_shared_head)
                ft_pre_shared_head = cac_mask1 * ft_pre_shared_head
            if self.caa is not None:
                ft_pre_shared_head = self.caa(ft_pre_shared_head, rois[:, 0], x[0])
            if self.sa1:
                sa_mask1 = self.sa1(ft_pre_shared_head)
                ft_pre_shared_head = sa_mask1 * ft_pre_shared_head

            ft_post_shared_head = self.shared_head(ft_pre_shared_head)

            if self.ffm is not None:
                main_ft = self.ffm(ft_post_shared_head, ft_pre_shared_head)
                det_feat = ft_post_shared_head
            else:
                main_ft = ft_post_shared_head

            # post id-net attention
            if self.cac2:
                cac_mask2 = self.cac2(main_ft)
                main_ft = cac_mask2 * main_ft
            if self.sa2:
                sa_mask2 = self.sa2(main_ft)
                main_ft = sa_mask2 * main_ft
        cls_score, bbox_pred, reid_feat, info_loss, reid_pre_proj = self.bbox_head(main_ft, det_feat,
                                                                                   fgbg_flag=fgbg_flag)

        if det_feat is None:
            det_feat = main_ft
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=det_feat, reid_feat=reid_feat, info_loss=info_loss,
            reid_pre_proj=reid_pre_proj)
        return bbox_results, sa_mask1, sa_mask2

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        fgbg_flag = bbox_targets[0] == 0
        gt_flag = bbox_targets[-1] == 1
        bbox_results, sa_mask1, sa_mask2 = self._bbox_forward(x, rois, fgbg_flag=fgbg_flag)

        # import ipdb; ipdb.set_trace()

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'],
                                        bbox_results['reid_feat'],
                                        bbox_results['reid_pre_proj'],
                                        rois,
                                        *bbox_targets)
        # attention loss
        loss_attention = dict()
        gt_bboxes = self.bbox_head.bbox_coder.decode(rois[:, 1:], bbox_targets[3])
        if self.att_loss_policy & 0x01:
            kwargs = self.margin_att_loss_param[0]
            if kwargs is None:
                kwargs = dict()
            loss = self.sa1.margin_loss(sa=sa_mask1, **kwargs)
            loss[~fgbg_flag] *= 0
            loss = torch.sum(loss)
            if torch.sum(fgbg_flag) > 0:
                loss /= torch.sum(fgbg_flag)
            loss_attention["att1_margin_loss"] = loss * self.att_loss_weight[0]
        if self.att_loss_policy & 0x02:
            loss = self.sa1.consistency_loss(sa_mask1)
            loss[~fgbg_flag] *= 0
            loss = torch.sum(loss)
            if torch.sum(fgbg_flag) > 0:
                loss /= torch.sum(fgbg_flag)
            loss_attention["att1_consis_loss"] = loss * self.att_loss_weight[1]
        if self.att_loss_policy & 0x04:
            kwargs = self.margin_att_loss_param[1]
            if kwargs is None:
                kwargs = dict()
            loss = self.sa2.margin_loss(sa=sa_mask2, **kwargs)
            loss[~fgbg_flag] *= 0
            loss = torch.sum(loss)
            if torch.sum(fgbg_flag) > 0:
                loss /= torch.sum(fgbg_flag)
            loss_attention["att2_margin_loss"] = loss * self.att_loss_weight[2]
        if self.att_loss_policy & 0x08:
            loss = self.sa2.consistency_loss(sa_mask2)
            loss[~fgbg_flag] *= 0
            loss = torch.sum(loss)
            if torch.sum(fgbg_flag) > 0:
                loss /= torch.sum(fgbg_flag)
            loss_attention["att2_consis_loss"] = loss * self.att_loss_weight[3]

        if self.att_loss_policy & 0x10:
            hatt1 = self.sa1.hard_attention_loss(sa_mask1, rois, gt_bboxes, fgbg_flag)
            if hatt1 is not None:
                loss_attention["hard_att1_loss"] = hatt1 * self.att_loss_weight[4]

        if self.att_loss_policy & 0x20:
            hatt2 = self.sa2.hard_attention_loss(sa_mask2, rois, gt_bboxes, fgbg_flag)
            if hatt2 is not None:
                loss_attention["hard_att2_loss"] = hatt2 * self.att_loss_weight[5]

        if self.relation_loss:
            total_relation_loss = []
            if self.relation_loss_policy & 0x01:
                total_relation_loss.append(
                    self.relation_loss(sa_mask1[gt_flag], gt_bboxes[gt_flag], rois[:, 0][gt_flag]))
            if self.relation_loss_policy & 0x02:
                total_relation_loss.append(
                    self.relation_loss(sa_mask2[gt_flag], gt_bboxes[gt_flag], rois[:, 0][gt_flag]))
            if len(total_relation_loss) > 0:
                loss_attention["relation_loss"] = sum(total_relation_loss) / len(total_relation_loss)
        bbox_results.update(loss_bbox=loss_bbox)
        bbox_results.update(loss_attention=loss_attention)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            if pos_rois.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            if pos_inds.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):  # changed!!!
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels, det_reid_feat = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)[0]
            for i in range(len(det_bboxes))
        ]
        # import ipdb; ipdb.set_trace()
        det_reid_feat = [each.cpu().numpy() for each in det_reid_feat]
        result = list(zip(bbox_results, det_reid_feat))
        result = [list(each) for each in result]
        # import ipdb; ipdb.set_trace()
        return result

    def simple_test_bboxes(self,  # from test_mixins
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        bbox_results, _, _ = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        reid_feat = bbox_results['reid_feat']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        reid_feat = reid_feat.split(num_proposals_per_img, 0)
        # some detector with_reg is False, bbox_pred will be None
        bbox_pred = bbox_pred.split(
            num_proposals_per_img,
            0) if bbox_pred is not None else [None, None]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        det_reid_feats = []
        for i in range(len(proposals)):
            det_bbox, det_label, det_reid_feat = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                reid_feat[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            det_reid_feats.append(det_reid_feat)
        # import ipdb; ipdb.set_trace()
        return det_bboxes, det_labels, det_reid_feats

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

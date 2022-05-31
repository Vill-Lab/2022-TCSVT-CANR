import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from torch.nn import init

from mmdet.core import (auto_fp16, force_fp32, multi_apply,
                        ps_multiclass_nms)
from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps
from mmdet.models.builder import HEADS
from mmdet.models.losses import ContrastivePersonSearchLoss
from mmdet.models.losses import OICLoss
from mmdet.models.losses import OIMLoss
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads.bbox_heads import BBoxHead


def entropy(var):
    return torch.mean(1 / 2 * (math.log(2 * math.pi) + 1) + 1 / 2 * torch.log(var), dim=1)


class VIB(nn.Module):
    def __init__(self, dim_in, dim_out, use_bn=False, mean_bias=False, beta=1e-3, mean_weight=0.5, kl_fg=False,
                 dbg=False, eps=1e-6, learnable_weight=False, vib_dropout=False, vib_dropout_p=0.5,
                 log_var=False, alt_info_loss=False, init=False, offset=5):
        super(VIB, self).__init__()
        self.eps = eps
        self.use_bn = use_bn
        self.mean_bias = mean_bias
        # print(dim_in, dim_out)
        self.proj_mean = nn.Sequential(nn.Linear(dim_in, dim_out, bias=self.mean_bias),
                                       nn.BatchNorm1d(dim_out, 2e-5)) if use_bn else nn.Linear(dim_in, dim_out,
                                                                                               bias=self.mean_bias)
        self.proj_var = nn.Sequential(nn.Linear(dim_in, dim_out),
                                      nn.BatchNorm1d(dim_out, 2e-5)) if use_bn else nn.Linear(dim_in, dim_out)
        self.norm_dist = torch.distributions.Normal(0.0, 1.0)
        self.beta = beta
        self.kl_fg = kl_fg
        self.dbg = dbg
        assert 0 <= mean_weight <= 1
        self.mean_weight = torch.nn.Parameter(
            torch.tensor(mean_weight, dtype=torch.float32)) if learnable_weight else mean_weight
        self.learnable_weight = learnable_weight
        self.dropout = torch.nn.Dropout(p=vib_dropout_p) if vib_dropout else None
        self.log_var = log_var
        self.alt_info_loss = alt_info_loss
        self.offset = offset
        if init:
            self._init_weight()
        print("USING VIB")

    def _init_weight(self):
        def _init_helper(module):
            init.normal_(module.weight, std=0.01)
            if module.bias is not None:
                init.constant_(module.bias, 0.0)

        if self.use_bn:
            for i in range(2):
                _init_helper(self.proj_mean[i])
                _init_helper(self.proj_var[i])
        else:
            _init_helper(self.proj_mean)
            _init_helper(self.proj_var)

    def _get_dist(self, mu, sigma):
        if self.log_var:
            ft_dist = torch.distributions.Normal(mu, torch.sqrt(
                torch.exp(sigma)))
        else:
            ft_dist = torch.distributions.Normal(mu, F.softplus(
                sigma - self.offset) + self.eps)
        return ft_dist

    def forward(self, in_ft, fgbg_flag=None):
        if self.training:
            ft_mean = self.proj_mean(in_ft)
            if self.dropout:
                ft_mean = self.dropout(ft_mean)
            ft_sigma = self.proj_var(in_ft)
            ft_dist = self._get_dist(ft_mean, ft_sigma)
            if self.mean_weight < 1e-3 and (not self.learnable_weight):
                ft = ft_dist.sample()
            else:
                ft = self.mean_weight * ft_mean + (1. - self.mean_weight) * ft_dist.sample()
            if self.kl_fg and fgbg_flag is not None:
                ft_dist_kl = self._get_dist(ft_mean[fgbg_flag], ft_sigma[fgbg_flag])
            else:
                ft_dist_kl = ft_dist
            if self.alt_info_loss:
                sigma_avg = 5
                threshold = np.log(sigma_avg) + (1 + np.log(2 * np.pi)) / 2
                info_loss = torch.mean(torch.relu(threshold - entropy(ft_dist_kl.scale))) * self.beta
            else:
                info_loss = torch.sum(
                    torch.mean(torch.distributions.kl_divergence(ft_dist_kl, self.norm_dist), dim=0)) * self.beta
            if self.dbg and torch.isinf(info_loss):
                import ipdb;
                ipdb.set_trace()
            return ft, info_loss
        else:
            ft = self.proj_mean(in_ft)
            return ft


class GradReverser(nn.Module):
    class ReverserFunction(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, lambda_):
            ctx.save_for_backward(lambda_)
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            lambda_, = ctx.saved_variables
            grad_input = grad_output.clone()
            return - lambda_ * grad_input, None

    def __init__(self, lambda_=0.):
        super(GradReverser, self).__init__()
        self.lambda_ = torch.tensor(lambda_)
        self.grf = self.ReverserFunction.apply

    def forward(self, x):
        return self.grf(x, self.lambda_)


class ScalePrediction(nn.Module):
    def __init__(self, dim_in, lambda_=0.1):
        super(ScalePrediction, self).__init__()
        self.fc1 = nn.Linear(dim_in, 16)
        self.bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 1)
        self.grl = GradReverser(lambda_)
        self.__init()

    def __init(self):
        torch.nn.init.constant_(self.fc2.bias, 9)

    def forward(self, x):
        x = self.grl(x)
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return x


class BorderAlignment(torch.nn.Module):
    def __init__(self, pool="avg"):
        super(BorderAlignment, self).__init__()
        # config for this module
        in_channels = 2048
        out_channels = 1024
        border_channels = 512
        self.border_channels = border_channels
        self.in2d = nn.InstanceNorm2d(border_channels * 4)
        self.pre_bd = nn.Sequential(
            nn.Conv2d(
                in_channels,
                border_channels * 4,
                kernel_size=1),
            nn.InstanceNorm2d(border_channels * 4),
            nn.ReLU())
        self.pool = pool

        # self.post_bd = nn.Linear(border_channels * 4, out_channels)
        # self.proj_ori = nn.Linear(in_channels, out_channels)

    def forward(self, x, o):
        # import ipdb; ipdb.set_trace()
        x = self.pre_bd(x)
        x = self.in2d(x)
        x = F.relu(x, inplace=True)
        # border max pooling, shape = (batch, channels)
        if self.pool == "max":
            ft_l = torch.max(x[:, :self.border_channels, :-1, 0], dim=2).values
            ft_t = torch.max(x[:, self.border_channels:self.border_channels * 2, 0, 1:], dim=2).values
            ft_r = torch.max(x[:, self.border_channels * 2:self.border_channels * 3, 1:, -1], dim=2).values
            ft_b = torch.max(x[:, self.border_channels * 3:, -1, :-1], dim=2).values
        else:
            assert self.pool == "avg"
            ft_l = torch.mean(x[:, :self.border_channels, :-1, 0], dim=2)
            ft_t = torch.mean(x[:, self.border_channels:self.border_channels * 2, 0, 1:], dim=2)
            ft_r = torch.mean(x[:, self.border_channels * 2:self.border_channels * 3, 1:, -1], dim=2)
            ft_b = torch.mean(x[:, self.border_channels * 3:, -1, :-1], dim=2)
        ft_border = torch.cat([ft_l, ft_t, ft_r, ft_b], dim=1)
        return ft_border


class DynamicCWS(nn.Module):

    def __init__(self, reid_dim, stop_grad=False):
        super(DynamicCWS, self).__init__()
        self.stop_grad = stop_grad
        self.fc1_reid = nn.Linear(reid_dim, 16)
        self.bn_reid = nn.BatchNorm1d(16)
        self.fc1_fgbg = nn.Linear(2, 16)
        self.bn_fgbg = nn.BatchNorm1d(16)
        self.fc_shared = nn.Linear(32, 1)

    def forward(self, reid_ft, fgbg_score):
        if self.stop_grad:
            reid_ft = reid_ft.detach()
            fgbg_score = fgbg_score.detach()
        ft = self.fc1_reid(reid_ft)
        ft = F.relu(self.bn_reid(ft), inplace=True)

        fgbg = self.fc1_fgbg(fgbg_score)
        fgbg = F.relu(self.bn_fgbg(fgbg), inplace=True)

        w = F.sigmoid(self.fc_shared(torch.cat([ft, fgbg], dim=1)))

        return w


@HEADS.register_module()
class PersonSearchBBoxHead(BBoxHead):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self, reid_dim=256, reid_bias=False, num_pid=5532, size_queue=5000, oim_momentum=0.5, oim_temp=10,
                 vib=False, vib_weight=0.001, arcface=None, ce="normal", bam=False, center_pooling=False, bampool="avg",
                 contrastive_loss=False, cont_scale_a=2.5, cont_scale_r=2.5, cont_scale_ax=5, cont_scale_rx=5,
                 cont_loss_policy=3, vib_bn=False, norm_type="BN1d", cont_lambda=0.1, oic_queue=0, oic_scale=10,
                 scale_pred_lam=None, gt_update_only=False, mean_weight=0.5, lut_inst=1, reid_ftbn=False,
                 reid_postbn=False, no_reid_proj=False, reid_max_pooling=False, weight_mat=None, loss_weight=None,
                 ff_policy=0, det_max_pooling=False, det_with_reid_ft=False, adaptive_scale=False, scale_multi=None,
                 cls_reg_bn=False, kl_fg=False, dbg=False, ff_scale=1.0, ce_threshold=0.75, cc_loss_cfg=None,
                 cc_logit_bn=False, w_rand=True, with_oim=True, with_ex_oim=False, with_ex_cc_loss=False, vib_eps=1e-6,
                 vib_learnable_weight=False, vib_dropout=False, vib_dropout_p=0.5, log_var=False, rand_th=None,
                 lbl_smoothing=0.1,
                 alt_info_loss=False, vib_offset=5,
                 vib_init=False, *args, **kwargs):
        super(PersonSearchBBoxHead, self).__init__(*args, **kwargs)
        self.num_pid = num_pid
        self.with_ex_cc_loss = with_ex_cc_loss
        self.with_cc_loss = contrastive_loss
        self.det_max_pooling = det_max_pooling
        self.det_with_reid_ft = det_with_reid_ft
        self.center_pooling = center_pooling
        self.reid_max_pooling = reid_max_pooling
        self.loss_weight = dict() if loss_weight is None else loss_weight
        self.oim = OIMLoss(num_pid=num_pid, size_queue=size_queue, reid_dim=reid_dim, momentum=oim_momentum,
                           temperature=oim_temp, ce=ce, gt_update_only=gt_update_only, lut_inst=lut_inst,
                           weight_mat=weight_mat, adaptive_scale=adaptive_scale, scale_multi=scale_multi,
                           ce_threshold=ce_threshold, rand=w_rand, rand_th=rand_th,
                           smoothing=lbl_smoothing) if with_oim else None

        self.ex_oim = OIMLoss(num_pid=num_pid, size_queue=size_queue, reid_dim=2048, momentum=oim_momentum,
                              temperature=oim_temp, ce=ce, gt_update_only=gt_update_only, lut_inst=lut_inst,
                              weight_mat=weight_mat, adaptive_scale=adaptive_scale, scale_multi=scale_multi,
                              ce_threshold=ce_threshold, rand=w_rand) if with_ex_oim else None

        assert not oic_queue > 0
        self.oic = OICLoss(size_queue=oic_queue, reid_dim=reid_dim, temperature=oic_scale,
                           num_pid=num_pid) if oic_queue > 0 else None
        self.use_vib = vib
        self.reid_bn = build_norm_layer({"type": norm_type, "eps": 2e-5}, 2048)[1] if reid_ftbn else None
        self.reid_postbn = build_norm_layer({"type": norm_type, "eps": 2e-5}, reid_dim)[1] if reid_postbn else None
        self.cls_reg_bn = build_norm_layer({"type": norm_type, "eps": 2e-5}, 2048)[1] if cls_reg_bn else None
        self.bam = BorderAlignment(pool=bampool) if bam else None
        self.ff_scale = ff_scale
        if ff_policy > 0:
            assert self.reid_max_pooling
        if ff_policy == 1:
            self.ff_weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float))
        elif ff_policy == 2:
            self.ff_weight = torch.nn.Parameter(torch.zeros(2048, dtype=torch.float))
        else:
            self.ff_weight = None
        self.contrastive_loss = ContrastivePersonSearchLoss(lambda_=cont_lambda, scale_a=cont_scale_a,
                                                            scale_r=cont_scale_r, scale_ax=cont_scale_ax,
                                                            scale_rx=cont_scale_rx,
                                                            loss_policy=cont_loss_policy,
                                                            dbg=dbg,
                                                            loss_cfg=cc_loss_cfg,
                                                            logit_bn=cc_logit_bn,
                                                            norm_type=norm_type) if contrastive_loss or with_ex_cc_loss else None
        if no_reid_proj:
            self.fc_feat_reid = None
        else:
            if not self.use_vib:
                self.fc_feat_reid = nn.Linear(self.in_channels, reid_dim, bias=reid_bias)
            else:
                self.fc_feat_reid = VIB(self.in_channels, reid_dim, mean_bias=reid_bias, beta=vib_weight,
                                        mean_weight=mean_weight, use_bn=vib_bn, kl_fg=kl_fg, dbg=dbg, eps=vib_eps,
                                        learnable_weight=vib_learnable_weight, vib_dropout=vib_dropout,
                                        vib_dropout_p=vib_dropout_p, log_var=log_var, init=vib_init,
                                        alt_info_loss=alt_info_loss, offset=vib_offset)

        assert scale_pred_lam is None or isinstance(scale_pred_lam, (float, bool))
        self.scale_pred = ScalePrediction(reid_dim, scale_pred_lam) if scale_pred_lam else None
        assert self.with_avg_pool

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    @auto_fp16()
    def forward(self, main_ft, det_ft=None, fgbg_flag=None):  # changed!
        batch_size = main_ft.size(0)
        assert self.with_avg_pool
        if self.center_pooling:
            to_pool_main = main_ft[..., 1:-1, 1:-1]
        else:
            to_pool_main = main_ft

        main_avg_pool = self.avg_pool(to_pool_main).view(batch_size, -1)
        if (det_ft is None and self.det_max_pooling) or self.reid_max_pooling:
            main_max_pool = torch.nn.functional.adaptive_max_pool2d(to_pool_main, 1).view(batch_size, -1)
        else:
            main_max_pool = None

        if self.reid_max_pooling:
            if self.ff_weight is None:
                reid_feat = main_max_pool
            else:
                lam = torch.sigmoid(self.ff_weight * self.ff_scale)
                reid_feat = main_avg_pool * lam + main_max_pool * (1 - lam)
        else:
            reid_feat = main_avg_pool

        # CLS&REG
        assert self.bam is None or det_ft is None  # bam is not compatible with dual head
        if self.bam:
            cls_reg_feat = self.bam(main_ft, main_avg_pool)
        elif det_ft is not None:  # dual head
            if self.det_max_pooling:
                cls_reg_feat = torch.nn.functional.adaptive_max_pool2d(det_ft, 1).view(-1, 2048)
            else:
                cls_reg_feat = torch.nn.functional.adaptive_avg_pool2d(det_ft, 1).view(-1, 2048)
        else:
            if self.det_with_reid_ft:
                cls_reg_feat = reid_feat
            else:
                if self.det_max_pooling:
                    cls_reg_feat = main_max_pool
                else:
                    cls_reg_feat = main_avg_pool

        if self.cls_reg_bn:
            cls_reg_feat = self.cls_reg_bn(cls_reg_feat)
        cls_score = self.fc_cls(cls_reg_feat) if self.with_cls else None
        bbox_pred = self.fc_reg(cls_reg_feat) if self.with_reg else None

        # REID
        info_loss = None  # vib loss
        if self.reid_bn is not None:
            reid_feat = self.reid_bn(reid_feat)
        reid_pre_proj = reid_feat
        # proj reid
        if self.fc_feat_reid is not None:
            if self.training and self.use_vib:
                reid_feat, info_loss = self.fc_feat_reid(reid_feat, fgbg_flag=fgbg_flag)
            else:
                reid_feat = self.fc_feat_reid(reid_feat)
            if self.reid_postbn:
                reid_feat = self.reid_postbn(reid_feat)
        reid_feat = F.normalize(reid_feat)

        return cls_score, bbox_pred, reid_feat, info_loss, reid_pre_proj  # return none when info_loss not exist

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, pos_is_gt, pos_is_jitter, cfg):  # changed!  # attention!!!
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # ORIGINAL ANNOTATION FROM MMDET:
        # # original implementation uses new_zeros since BG are set to be 0
        # # now use empty & fill because BG cat_id = num_classes,
        # # FG cat_id = [0, num_classes-1]

        # for person search:
        # FG with identity label: [0, num_pid - 1]
        # FG without identity label: < 0
        # BG: num_pid + 1
        # import ipdb; ipdb.set_trace()
        labels = pos_bboxes.new_full((num_samples,), self.num_pid + 1,
                                     dtype=torch.long)  # neg = num_pos + 1
        pos_flags = pos_bboxes.new_full((num_pos,), 0, dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        bbox_iou = pos_bboxes.new_zeros(num_samples)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_flags[pos_is_gt] += 1
            pos_flags[pos_is_jitter] += 2
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            bbox_iou[:num_pos] = bbox_overlaps(pos_bboxes, pos_gt_bboxes, is_aligned=True)  # to debug
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0
        fgbg_labels = torch.zeros_like(labels, dtype=labels.dtype)
        fgbg_labels[labels > self.num_pid] = 1

        flags_ = pos_bboxes.new_full((num_samples,), 0, dtype=torch.long)
        flags_[:num_pos] = pos_flags

        return fgbg_labels, labels, label_weights, bbox_targets, bbox_weights, bbox_iou, flags_

    def get_targets(self,  # TODO
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        # import ipdb; ipdb.set_trace()
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_is_gt_list = [res.pos_is_gt > 0 for res in sampling_results]
        pos_is_jitter_list = [res.pos_inds > 2000 for res in sampling_results]  # TODO: hard coded, to be improved later
        # import ipdb; ipdb.set_trace()
        fgbg_labels, labels, label_weights, bbox_targets, bbox_weights, bbox_iou, flags_ = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            pos_is_gt_list,
            pos_is_jitter_list,
            cfg=rcnn_train_cfg)

        if concat:
            fgbg_labels = torch.cat(fgbg_labels, 0)
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            bbox_iou = torch.cat(bbox_iou, 0)
            flags_ = torch.cat(flags_, 0)

        return fgbg_labels, labels, label_weights, bbox_targets, bbox_weights, bbox_iou, flags_

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,  # changed
             cls_score,
             bbox_pred,
             reid_feat,
             reid_pre_proj,
             rois,
             fgbg_label,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             bbox_iou,
             flags_,
             reduction_override=None):
        losses = dict()
        if self.oim is not None:
            losses["oim_scale"] = self.oim.temperature
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    fgbg_label,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, fgbg_label)
        if bbox_pred is not None:
            pos_inds = fgbg_label == 0
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           fgbg_label[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
        if reid_feat is not None and self.oim is not None:
            losses["loss_reid"] = self.oim(reid_feat, labels, flags_)
        if reid_pre_proj is not None and self.ex_oim is not None:
            losses["loss_ex_reid"] = self.ex_oim(F.normalize(reid_pre_proj), labels, flags_)
        if self.with_ex_cc_loss:
            ex_cc_loss = self.contrastive_loss(F.normalize(reid_pre_proj), labels, flags_, fgbg_label, rois)
            for k, v in ex_cc_loss.items():
                losses["ex_" + k] = v
        if self.with_cc_loss:
            cc_losses = self.contrastive_loss(reid_feat, labels, flags_, fgbg_label, rois)
            losses.update(cc_losses)
        if self.oic:
            losses["oic_loss"] = self.oic(reid_feat, labels)
        if self.scale_pred is not None:
            pred_scale = self.scale_pred(reid_feat).squeeze(1)
            targets = torch.log((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
            valid = targets > 5
            pred_scale = pred_scale[valid]
            targets = targets[valid]
            losses["scale_adv_loss"] = torch.nn.functional.l1_loss(pred_scale, targets)

        for k, v in self.loss_weight.items():
            if k in losses.keys():
                losses[k] *= v

        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   reid_feat,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            # import ipdb; ipdb.set_trace()
            det_bboxes, det_labels, det_reid_feat = ps_multiclass_nms(bboxes, scores, reid_feat,
                                                                      cfg.score_thr, cfg.nms,
                                                                      cfg.max_per_img)

            return det_bboxes, det_labels, det_reid_feat

    @force_fp32(apply_to=('bbox_preds',))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image. The first column is
                the image id and the next 4 columns are x1, y1, x2, y2.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwarray)
            >>> import kwarray
            >>> import numpy as np
            >>> from mmdet.core.bbox.demodata import random_boxes
            >>> self = BBoxHead(reg_class_agnostic=True)
            >>> n_roi = 2
            >>> n_img = 4
            >>> scale = 512
            >>> rng = np.random.RandomState(0)
            >>> img_metas = [{'img_shape': (scale, scale)}
            ...              for _ in range(n_img)]
            >>> # Create rois in the expected format
            >>> roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
            >>> img_ids = torch.randint(0, n_img, (n_roi,))
            >>> img_ids = img_ids.float()
            >>> rois = torch.cat([img_ids[:, None], roi_boxes], dim=1)
            >>> # Create other args
            >>> labels = torch.randint(0, 2, (n_roi,)).long()
            >>> bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
            >>> # For each image, pretend random positive boxes are gts
            >>> is_label_pos = (labels.numpy() > 0).astype(np.int)
            >>> lbl_per_img = kwarray.group_items(is_label_pos,
            ...                                   img_ids.numpy())
            >>> pos_per_img = [sum(lbl_per_img.get(gid, []))
            ...                for gid in range(n_img)]
            >>> pos_is_gts = [
            >>>     torch.randint(0, 2, (npos,)).byte().sort(
            >>>         descending=True)[0]
            >>>     for npos in pos_per_img
            >>> ]
            >>> bboxes_list = self.refine_bboxes(rois, labels, bbox_preds,
            >>>                    pos_is_gts, img_metas)
            >>> print(bboxes_list)
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred',))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=img_meta['img_shape'])
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois

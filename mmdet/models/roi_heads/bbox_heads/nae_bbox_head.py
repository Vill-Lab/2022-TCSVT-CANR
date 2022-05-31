import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import (auto_fp16, force_fp32, multi_apply,
                        ps_multiclass_nms)
from mmdet.models.builder import HEADS
from mmdet.models.losses import OIMLoss
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads.bbox_heads import BBoxHead
import ipdb
from mmcv.cnn import build_norm_layer


from torch.nn import init


class VIB(nn.Module):
    def __init__(self, dim_in, dim_out, use_bn=False, mean_bias=False, beta=1e-3):
        super(VIB, self).__init__()
        self.use_bn = use_bn
        self.mean_bias = mean_bias
        # print(dim_in, dim_out)
        self.proj_mean = nn.Sequential(nn.Linear(dim_in, dim_out, bias=self.mean_bias),
                                       nn.BatchNorm1d(dim_out, 2e-5)) if use_bn else nn.Linear(dim_in, dim_out, bias=self.mean_bias)
        self.proj_var = nn.Sequential(nn.Linear(dim_in, dim_out),
                                      nn.BatchNorm1d(dim_out, 2e-5)) if use_bn else nn.Linear(dim_in, dim_out)
        self.norm_dist = torch.distributions.Normal(0.0, 1.0)
        self.beta = beta
        # self._init_weight()
        print("USING VIB")

    def _init_weight(self):
        if self.use_bn:
            for i in range(2):
                self._init_helper(self.proj_mean[i])
                self._init_helper(self.proj_var[i])
        else:
            self._init_helper(self.proj_mean)
            self._init_helper(self.proj_var)

    def _init_helper(self, module):
        init.normal_(module.weight, std=0.01)
        if module.bias is not None:
            init.constant_(module.bias, 0.0)


    def forward(self, in_ft):
        if self.training:
            ft_mean = self.proj_mean(in_ft)
            ft_var = self.proj_var(in_ft)
            ft_dist = torch.distributions.Normal(ft_mean, F.softplus(ft_var - 5))
            ft = ft_mean + ft_dist.sample()
            info_loss = torch.sum(torch.mean(torch.distributions.kl_divergence(ft_dist, self.norm_dist),dim=0)) * self.beta
            return ft, info_loss
        else:
            ft = self.proj_mean(in_ft)
            return ft



@HEADS.register_module()
class NAEPersonSearchBBoxHead(BBoxHead):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self, reid_dim=256, reid_bias=False, num_pid=5532, size_queue=5000, oim_momentum=0.5, oim_temp=10, vib=False, vib_weight=0.001, rcnn_box_bn=False, ft_bn=False,
                 *args, **kwargs):
        super(NAEPersonSearchBBoxHead, self).__init__(*args, with_cls=False, **kwargs)
        self.num_pid = num_pid
        self.oim = OIMLoss(num_pid=num_pid, size_queue=size_queue, reid_dim=reid_dim, momentum=oim_momentum, temperature=oim_temp)
        self.use_vib = vib
        self.rescaler = build_norm_layer(dict(type='SyncBN'),1)[1]
        self.rcnn_box_bn = build_norm_layer(dict(type='SyncBN'),4 if self.reg_class_agnostic else 4 * self.num_classes)[1] if rcnn_box_bn else None
        if not self.use_vib:
            self.fc_feat_reid = nn.Linear(self.in_channels, reid_dim, bias=reid_bias)
        else:
            self.fc_feat_reid = VIB(self.in_channels, reid_dim, mean_bias=reid_bias, beta=vib_weight)
        self.ft_bn = build_norm_layer(dict(type='SyncBN'),reid_dim)[1] if ft_bn else None

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    @auto_fp16()
    def forward(self, x):  # WIP
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        if self.rcnn_box_bn:
            bbox_pred = self.rcnn_box_bn(bbox_pred)
        info_loss = None  # vib loss
        if self.training:
            if self.use_vib:
                reid_feat, info_loss = self.fc_feat_reid(x)
            else:
                reid_feat = self.fc_feat_reid(x)
        else:  # eval
            reid_feat = self.fc_feat_reid(x)
        if self.ft_bn:
            reid_feat = self.ft_bn(reid_feat)
        norms = torch.norm(reid_feat, dim=1, keepdim=True)
        cls_score = self.rescaler(norms)
        if self.training:
            reid_feat = F.normalize(reid_feat)
        else:
            reid_feat = F.normalize(reid_feat) * norms

        return cls_score, bbox_pred, reid_feat, info_loss  # return none when info_loss not exist

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):  # changed!  # attention!!!
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # ORIGINAL ANNOTATION FROM MMDET:
        # # original implementation uses new_zeros since BG are set to be 0
        # # now use empty & fill because BG cat_id = num_classes,
        # # FG cat_id = [0, num_classes-1]

        # for person search:
        # FG with identity label: [0, num_pid - 1]
        # FG without identity label: num_pid
        # BG: num_pid + 1
        # ipdb.set_trace()
        labels = pos_bboxes.new_full((num_samples,),
                                     self.num_pid + 1,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
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

        return fgbg_labels, labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,  # todo
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        fgbg_labels, labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            fgbg_labels = torch.cat(fgbg_labels, 0)
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)

        return fgbg_labels, labels, label_weights, bbox_targets, bbox_weights

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self, # changed
             cls_score,
             bbox_pred,
             reid_feat,
             rois,
             fgbg_label,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    1 - fgbg_label,  # because [0, n - 1] is forground label and n is background label
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, 1 - fgbg_label)
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
        if reid_feat is not None:
            losses["loss_reid"] = self.oim(reid_feat, labels)
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
        cls_score = torch.sigmoid(cls_score)
        cls_score = torch.cat([cls_score, 1-cls_score], dim=-1)

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

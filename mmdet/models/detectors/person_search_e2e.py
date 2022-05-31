from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os.path as osp


@DETECTORS.register_module()
class PersonSearchE2E(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 extra_loss_weight=None):
        super(PersonSearchE2E, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        # self.cnter = 0
        self.extra_loss_weight = dict() if extra_loss_weight is None else extra_loss_weight

    def forward_train(self, *args, **kwargs):
        losses = super(PersonSearchE2E, self).forward_train(*args, **kwargs)
        for k in losses.keys():
            if k in self.extra_loss_weight:
                losses[k] *= self.extra_loss_weight[k]
        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)
        # outputs = x[0]
        # outputs = (outputs ** 2).sum(1)
        # b, h, w = outputs.size()
        # outputs = outputs.view(b, h * w)
        # outputs = F.normalize(outputs, p=2, dim=1)
        # outputs = outputs.view(b, h, w)
        # am = outputs[0].cpu().numpy()
        # am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
        # am = np.uint8(np.floor(am))
        # am = cv2.applyColorMap(am, cv2.COLORMAP_JET)
        # cv2.imwrite(osp.join("/home/chenzhicheng/GitRepo/mmdetection/scene_am", "{:05d}".format(self.cnter) + '_canrp.jpg'), am)
        # self.cnter += 1

        # import ipdb; ipdb.set_trace()

        if proposals is None or torch.norm(proposals).item() < 1e-3:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
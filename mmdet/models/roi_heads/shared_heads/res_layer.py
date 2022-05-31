import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from mmdet.core import auto_fp16
from mmdet.models.backbones import ResNet
from mmdet.models.builder import SHARED_HEADS
from mmdet.models.utils import ResLayer as _ResLayer
from mmdet.models.utils import ResLayerPS
from mmdet.utils import get_root_logger


@SHARED_HEADS.register_module()
class ResLayer(nn.Module):

    def __init__(self,
                 depth,
                 stage=3,
                 stride=2,
                 dilation=1,
                 style='pytorch',
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 with_cp=False,
                 dcn=None):
        super(ResLayer, self).__init__()
        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.stage = stage
        self.fp16_enabled = False
        block, stage_blocks = ResNet.arch_settings[depth]
        stage_block = stage_blocks[stage]
        planes = 64 * 2**stage
        inplanes = 64 * 2**(stage - 1) * block.expansion

        res_layer = _ResLayer(
            block,
            inplanes,
            planes,
            stage_block,
            stride=stride,
            dilation=dilation,
            style=style,
            with_cp=with_cp,
            norm_cfg=self.norm_cfg,
            dcn=dcn)
        self.add_module(f'layer{stage + 1}', res_layer)

    def init_weights(self, pretrained=None):
        """Initialize the weights in the module.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    @auto_fp16()
    def forward(self, x):
        res_layer = getattr(self, f'layer{self.stage + 1}')
        out = res_layer(x)
        return out

    def train(self, mode=True):
        super(ResLayer, self).train(mode)
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()




@SHARED_HEADS.register_module()
class ResLayers(nn.Module):

    def __init__(self,
                 depth,
                 stage=3,
                 stride=2,
                 dilation=1,
                 style='pytorch',
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 with_cp=False,
                 dcn=None,
                 offset=(0,),
                 end_trim = None):
        super(ResLayers, self).__init__()
        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.stages = stage if isinstance(stage, (list, tuple)) else [stage]
        self.fp16_enabled = False
        block, stage_blocks = ResNet.arch_settings[depth]
        if not isinstance(stride, (list, tuple)):
            stride = [stride for _ in self.stages ]
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation for _ in self.stages ]
        if end_trim is None:
            end_trim = [0 for _ in range(len(self.stages))]
        for i, stage in enumerate(self.stages):
            stage_block = stage_blocks[stage]
            planes = 64 * 2**stage
            inplanes = 64 * 2**(stage - 1) * block.expansion

            res_layer = ResLayerPS(
                block,
                inplanes,
                planes,
                stage_block,
                stride=stride[i],
                dilation=dilation[i],
                style=style,
                with_cp=with_cp,
                norm_cfg=self.norm_cfg,
                dcn=dcn,
                offset=offset[i],
                end_trim=end_trim[i])
            self.add_module(f'layer{stage + 1}', res_layer)

    def init_weights(self, pretrained=None):
        """Initialize the weights in the module.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    @auto_fp16()
    def forward(self, x):
        for stage in self.stages:
            res_layer = getattr(self, f'layer{stage + 1}')
            x = res_layer(x)
        return x

    def train(self, mode=True):
        super(ResLayers, self).train(mode)
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


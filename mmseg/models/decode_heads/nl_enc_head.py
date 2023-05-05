# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import NonLocal2d

from ..builder import HEADS
from .fcn_head import FCNHead
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmseg.ops import resize


@HEADS.register_module()
class NLEncHead(FCNHead):
    """Non-local Neural Networks.

    This head is the implementation of `NLNet
    <https://arxiv.org/abs/1711.07971>`_.

    Args:
        reduction (int): Reduction factor of projection transform. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            sqrt(1/inter_channels). Default: True.
        mode (str): The nonlocal mode. Options are 'embedded_gaussian',
            'dot_product'. Default: 'embedded_gaussian.'.
    """

    def __init__(self,
                 c1_in_channels,
                 c1_channels,
                 reduction=2,
                 use_scale=True,
                 mode='embedded_gaussian',
                 **kwargs):
        super(NLEncHead, self).__init__(num_convs=2, **kwargs)
        self.reduction = reduction
        self.use_scale = use_scale
        self.mode = mode
        self.nl_block = NonLocal2d(
            in_channels=self.channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode)

        self.c1_bottleneck = ConvModule(
            c1_in_channels,
            c1_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

        
    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs[0](x)
        output = self.nl_block(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        
        c1_output = self.c1_bottleneck(inputs[1])
        output = resize(
            input=output,
            size=c1_output.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        output = self.cls_seg(output)
        return output

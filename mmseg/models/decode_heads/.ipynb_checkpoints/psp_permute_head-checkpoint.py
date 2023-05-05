# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


# class permute_module(nn.Module):
#     """Permutation applied in PPM_permute
    
#     Args:
#         pool_scale(int): Pooling scale in this branch.
#         in_channels(int):Input channels
#         channels(int):Hidden channels. Need to be devided by spatial area(pool_scale**2).
#         conv_cfg (dict|None): Config of conv layers.
#         norm_cfg (dict|None): Config of norm layers.
#         act_cfg (dict): Config of activation layers.
#     """
#     def __init__(self, pool_scale, in_channels, channels, conv_cfg, norm_cfg,
#                  act_cfg, **kwargs):
#         super(permute_module, self).__init__()
        
#         self.pool_scale = pool_scale
#         self.in_channels = in_channels
#         self.channels = channels
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#         self.act_cfg = act_cfg

#         self.global_pool = nn.AdaptiveAvgPool2d(pool_scale)
#         self.in_conv = ConvModule(
#             self.in_channels,
#             self.channels,
#             1,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg,
#             **kwargs)
#         self.perm_conv = ConvModule(
#             self.channels,
#             self.channels,
#             1,
#             conv_cfg=self.conv_cfg,
#             norm_cfg=self.norm_cfg,
#             act_cfg=self.act_cfg,
#             **kwargs)
#     def forward(self, x):
#         x = self.global_pool(x)
#         x = self.in_conv(x) # bs * channels * pool_scale * pool_scale

#         b, c, h, w = x.shape
#         # print(x.shape)
#         x = x.reshape(b, c // (h * w), h * w, h * w).permute(0, 1, 3, 2).reshape(b, c, h, w)
#         x = self.perm_conv(x)

#         x = x.reshape(b, c // (h * w), h * w, h * w).permute(0, 1, 3, 2).reshape(b, c, h, w)
        
#         return x

class permute_module(nn.Module):
    """Permutation applied in PPM_permute
    
    Args:
        pool_scale(int): Pooling scale in this branch.
        in_channels(int):Input channels
        channels(int):Hidden channels. Need to be devided by spatial area(pool_scale**2).
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """
    def __init__(self, pool_scale, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, **kwargs):
        super(permute_module, self).__init__()
        
        self.pool_scale = pool_scale
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.global_pool = nn.AdaptiveAvgPool2d(pool_scale)
        self.in_conv = ConvModule(
            self.in_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            **kwargs)
        
        num_groups = channels // (pool_scale * pool_scale)
        self.perm_conv = ConvModule(
            self.channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            groups=num_groups,
            **kwargs)

    def forward(self, x):
        x = self.global_pool(x)
        x = self.in_conv(x) # bs * channels * pool_scale * pool_scale

        b, c, h, w = x.shape
        # print(x.shape)
        x = x.reshape(b, c // (h * w), h * w, h * w).permute(0, 1, 3, 2).reshape(b, c, h, w)
        x1 = self.perm_conv(x)
        x = x1.sigmoid() * x + x
        x = x.reshape(b, c // (h * w), h * w, h * w).permute(0, 1, 3, 2).reshape(b, c, h, w)
        
        return x
    
class PermutePPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet with Permutation

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channel_nums, conv_cfg, norm_cfg,
                 act_cfg, align_corners, **kwargs):
        super(PermutePPM, self).__init__()
        
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channel_nums = channel_nums
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.total_channels = self.in_channels
        if isinstance(pool_scales, int):
            pool_scale = pool_scales
            channels = channel_nums
            self.append(
                permute_module(pool_scale, in_channels, channels, conv_cfg, norm_cfg, act_cfg))
            self.total_channels += channels
        else:
            for pool_scale, channels in zip(pool_scales, channel_nums):
                self.append(
                    permute_module(pool_scale, in_channels, channels, conv_cfg, norm_cfg, act_cfg))
                self.total_channels += channels

    def forward(self, x):
        """Forward function."""
        ppm_p_outs = [x]
        for ppm_p in self:
            ppm_p_out = ppm_p(x)
            upsampled_ppm_p_out = resize(
                ppm_p_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_p_outs.append(upsampled_ppm_p_out)
        return ppm_p_outs
    
    
@HEADS.register_module()
class PSPPermuteHead(BaseDecodeHead):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """
    def __init__(self, pool_scales=(1, 6, 8, 12), channel_nums=(512, 540, 512, 576), interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)
        
        self.total_inchannel = 0
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.total_inchannel += self.in_channels[i]
                    
        self.ppm = PermutePPM(pool_scales=pool_scales, in_channels=self.total_inchannel,
        channels=self.channels, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, channel_nums=channel_nums,
        align_corners=self.align_corners)

        self.fusion_conv = ConvModule(
            in_channels=self.ppm.total_channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            outs.append(
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        outs = self.ppm(torch.cat(outs, dim=1))
        out = self.fusion_conv(torch.cat(outs, dim=1))

        # print(out.shape)
        out = self.cls_seg(out)
        
        return out
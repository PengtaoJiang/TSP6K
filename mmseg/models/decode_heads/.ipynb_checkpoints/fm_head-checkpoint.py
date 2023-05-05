# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import math
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F
import torch.nn.init as init

from mmseg.ops import Upsample, resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class FMBlock(nn.ModuleList):
    def __init__(self, in_channels=512, fm_channels=512, window_size=3):
        super(FMBlock, self).__init__()
        self.in_channels = in_channels
        self.fm_channels = fm_channels
        self.window_size = window_size
        
        self.down = nn.Conv2d(in_channels, fm_channels, 1, 1, 0, bias=True)
        self.conv1_f = nn.Conv2d(fm_channels, 2 * fm_channels, 3, 1, 1, bias=True)
        self.conv1_w = nn.Conv2d(fm_channels, fm_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(fm_channels, fm_channels, 3, 1, 1, bias=True)
        self.up = nn.Conv2d(fm_channels, in_channels, 1, 1, 0, bias=True)
        
        self.pi = 3.1415927410125732
        p = torch.arange(0, window_size, 1)
        q = torch.arange(0, window_size, 1)
        p, q = torch.meshgrid(p, q)
        self.p = ((p + 0.5)* self.pi / window_size).cuda().view(1, 1, 1, window_size, window_size)
        self.q = ((q + 0.5)* self.pi / window_size).cuda().view(1, 1, 1, window_size, window_size)


    def forward(self, x):
        N, C, H, W = x.shape
        K = self.fm_channels

        identity = x
        
        x = self.down(x)
        frequency = self.conv1_f(x) # N C H W -> N 2*K H W 
        weight = self.conv1_w(x) # N C H W -> N K H W 
        frequency = torch.sigmoid(frequency)
        weight = F.softmax(weight, dim=1)
        frequency = frequency * (self.window_size - 1)
        frequency = frequency.permute(0, 2, 3, 1).contiguous().view(N, H * W, 2, K) # N 2*K H W -> N H*W 2 K
        weight = weight.permute(0, 2, 3, 1).contiguous().view(N, H * W, 1, K) # N K H W -> N H*W 1 K
        hFrequency = frequency[:, :, 0, :].view(N , H * W, K, 1, 1).expand([-1, -1, -1, self.window_size, self.window_size])
        wFrequency = frequency[:, :, 1, :].view(N , H * W, K, 1, 1).expand([-1, -1, -1, self.window_size, self.window_size])
        p = self.p.expand([N, H * W, K, -1, -1])
        q = self.q.expand([N, H * W, K, -1, -1])
        
        kernel = torch.cos(hFrequency * p) * torch.cos(wFrequency * q)
        kernel = kernel.view(N, H * W, K, self.window_size ** 2)
        kernel = torch.matmul(weight, kernel)
        kernel = kernel.view(N, H * W, self.window_size ** 2, 1)
        # N H*W K**2 1

        v = F.unfold(x, kernel_size=self.window_size, padding=int((self.window_size - 1) / 2), stride=1) # N C H W -> N C*(K**2) H*W
        v = v.view(N, self.fm_channels, self.window_size ** 2, H * W) # N C K**2 H*W
        v = v.permute(0, 3, 1, 2).contiguous() # N H*W C K**2

        z = torch.matmul(v, kernel) # N H*W C 1
        z = z.squeeze(-1).view(N, H, W, self.fm_channels).permute(0, 3, 1, 2) # N H*W C -> N C H W
        z = self.conv2(z)
        out = self.up(z) 
        
        return identity + out

    def build_filter(self, kernelSize):
        filters = torch.zeros((kernelSize, kernelSize, kernelSize, kernelSize))
        for i in range(kernelSize):
            for j in range(kernelSize):
                for h in range(kernelSize):
                    for w in range(kernelSize):
                        filters[i, j, h, w] = math.cos(math.pi * i * (h + 0.5) / kernelSize) * math.cos(math.pi * j * (w + 0.5) / kernelSize)
        return filters.view(kernelSize ** 2, kernelSize, kernelSize).cuda()

    
@HEADS.register_module()
class FMHead(BaseDecodeHead):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self, fm_channels, window_size, **kwargs):
        super(FMHead, self).__init__(**kwargs)
        
        self.conv_dim = ConvModule(self.in_channels,
                self.channels, kernel_size=3, padding=1, dilation=1,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.fm_block = FMBlock(self.channels, fm_channels=fm_channels, window_size=window_size)
        
    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.conv_dim(x)
        output = self.fm_block(output)
        output = self.cls_seg(output)
        return output
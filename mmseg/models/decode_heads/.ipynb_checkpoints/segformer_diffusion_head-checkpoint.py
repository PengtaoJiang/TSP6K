# Modified from
# https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py
#
# This work is licensed under the NVIDIA Source Code License.
#
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License for StyleGAN2 with Adaptive Discriminator
# Augmentation (ADA)
#
#  1. Definitions
#  "Licensor" means any person or entity that distributes its Work.
#  "Software" means the original work of authorship made available under
# this License.
#  "Work" means the Software and any additions to or derivative works of
# the Software that are made available under this License.
#  The terms "reproduce," "reproduction," "derivative works," and
# "distribution" have the meaning as provided under U.S. copyright law;
# provided, however, that for the purposes of this License, derivative
# works shall not include works that remain separable from, or merely
# link (or bind by name) to the interfaces of, the Work.
#  Works, including the Software, are "made available" under this License
# by including in or with the Work either (a) a copyright notice
# referencing the applicability of this License to the Work, or (b) a
# copy of this License.
#  2. License Grants
#      2.1 Copyright Grant. Subject to the terms and conditions of this
#     License, each Licensor grants to you a perpetual, worldwide,
#     non-exclusive, royalty-free, copyright license to reproduce,
#     prepare derivative works of, publicly display, publicly perform,
#     sublicense and distribute its Work and any resulting derivative
#     works in any form.
#  3. Limitations
#      3.1 Redistribution. You may reproduce or distribute the Work only
#     if (a) you do so under this License, (b) you include a complete
#     copy of this License with your distribution, and (c) you retain
#     without modification any copyright, patent, trademark, or
#     attribution notices that are present in the Work.
#      3.2 Derivative Works. You may specify that additional or different
#     terms apply to the use, reproduction, and distribution of your
#     derivative works of the Work ("Your Terms") only if (a) Your Terms
#     provide that the use limitation in Section 3.3 applies to your
#     derivative works, and (b) you identify the specific derivative
#     works that are subject to Your Terms. Notwithstanding Your Terms,
#     this License (including the redistribution requirements in Section
#     3.1) will continue to apply to the Work itself.
#      3.3 Use Limitation. The Work and any derivative works thereof only
#     may be used or intended for use non-commercially. Notwithstanding
#     the foregoing, NVIDIA and its affiliates may use the Work and any
#     derivative works commercially. As used herein, "non-commercially"
#     means for research or evaluation purposes only.
#      3.4 Patent Claims. If you bring or threaten to bring a patent claim
#     against any Licensor (including any claim, cross-claim or
#     counterclaim in a lawsuit) to enforce any patents that you allege
#     are infringed by any Work, then your rights under this License from
#     such Licensor (including the grant in Section 2.1) will terminate
#     immediately.
#      3.5 Trademarks. This License does not grant any rights to use any
#     Licensor’s or its affiliates’ names, logos, or trademarks, except
#     as necessary to reproduce the notices described in this License.
#      3.6 Termination. If you violate any term of this License, then your
#     rights under this License (including the grant in Section 2.1) will
#     terminate immediately.
#  4. Disclaimer of Warranty.
#  THE WORK IS PROVIDED "AS IS" WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR
# NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER
# THIS LICENSE.
#  5. Limitation of Liability.
#  EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL
# THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE
# SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT,
# INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF
# OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK
# (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION,
# LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER
# COMMERCIAL DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGES.

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F 

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
import numpy as np
import math 

@HEADS.register_module()
class SegformerDiffusionHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        ## diffusion
        self.new_diffusion_convs = nn.ModuleList()
        for i in range(2):
            self.new_diffusion_convs.append(
                ConvModule(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        
        self.num_timesteps = 1000
        self.scale = 1.0

        betas = self.cosine_beta_schedule(self.num_timesteps)
        alphas = 1. - betas
        # self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        # self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.))

        # self.sqrt_alphas_cumprod =  torch.sqrt(self.alphas_cumprod)
        # self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))
        
        self.inference_steps = 10
        self.inference_stride = 1


    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self(inputs, gt_semantic_seg)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses
    
    def extract(self, a, t):
        return a.gather(0, t)
    
    def d_sample(self, targets, output_stride=4):
        with torch.no_grad():
            B, _, H, W = targets.shape
            targets = F.interpolate(targets.float(), (H//output_stride, W//output_stride), mode='nearest')
            targets = targets.squeeze(1).long()
            B, H, W = targets.shape

            targets_ont_hot_shape = (B, self.num_classes, H, W)
            targets_ont_hot = targets.new_zeros(targets_ont_hot_shape)
            valid_mask = (targets >= 0) & (targets != 255)
            inds = torch.nonzero(valid_mask, as_tuple=True)
            targets_ont_hot[inds[0], targets[valid_mask], inds[1], inds[2]] = 1

            noise_targets = (targets_ont_hot * 2 - 1) * self.scale # B, C, H, W
            t = torch.randint(0, self.num_timesteps, (B,), device=noise_targets.device).long()
            noise = torch.randn(targets_ont_hot_shape, device=noise_targets.device) # B, C, H, W

            alpha = self.extract(self.sqrt_alphas_cumprod, t).view(B, 1, 1, 1)
            alpha_1 = self.extract(self.sqrt_one_minus_alphas_cumprod, t).view(B, 1, 1, 1)
            noise_targets = alpha * noise_targets + alpha_1 * noise 

        return noise_targets

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def forward(self, inputs, gt_semantic_seg):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        ## diffusion
        noise_targets = self.d_sample(gt_semantic_seg).unsqueeze(2).float() # B, C, 1, H, W
        out1 = (noise_targets * out.unsqueeze(1)).sum(1) # B, C, H, W
        
        for i in range(2):
            out1 = self.new_diffusion_convs[i](out1)
        out = self.cls_seg(out1)

        return out


    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        ## diffusion
        B, C, H, W = out.shape
        out_T = torch.randn((B, self.num_classes, H, W), device=out.device)
        
        for step in range(self.inference_steps):
            t_now = (self.inference_steps - step) * 100 - 1
            t_next = max((self.inference_steps - step - 1) * 100, 0)

            t_now = torch.zeros((B,), device=out.device).fill_(t_now).long()
            t_next = torch.zeros((B,), device=out.device).fill_(t_next).long()

            alpha_t = self.extract(self.alphas_cumprod, t_now) 
            alpha_tprev = self.extract(self.alphas_cumprod, t_next) 
            
            out1 = (out_T.unsqueeze(2) * out.unsqueeze(1)).sum(1).float() # B, C, H, W
            for i in range(2):
                out1 = self.new_diffusion_convs[i](out1)
            out1 = self.cls_seg(out1)
            
            out_pred = torch.softmax(out1, dim=1).detach()
            eps = 1/torch.sqrt(1-alpha_t) * (out_T - torch.sqrt(alpha_t) * out_pred)
            out_T = torch.sqrt(alpha_tprev) * out_pred + torch.sqrt(1 - alpha_tprev) * eps
            
        return out1


# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
import torch.utils.checkpoint as cp
from mmcv.runner import ModuleList
from mmcv.cnn import ConvModule

# from mmseg.models.backbones.vit import TransformerEncoderLayer
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, attn_mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        if attn_mask is not None:
            attn = attn * attn_mask.unsqueeze(2)
            v = v * attn_mask.unsqueeze(3)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class TransformerMaskEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 attn_cfg=dict(),
                 ffn_cfg=dict(),
                 with_cp=False):
        super(TransformerMaskEncoderLayer, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        attn_cfg.update(
            dict(
                dim=embed_dims,
                heads=num_heads,
                dropout=attn_drop_rate))

        self.build_attn(attn_cfg)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        ffn_cfg.update(
            dict(
                dim=embed_dims,
                hidden_dim=embed_dims))
        self.build_ffn(ffn_cfg)
        self.with_cp = with_cp

    def build_attn(self, attn_cfg):
        self.attn = Attention(**attn_cfg)

    def build_ffn(self, ffn_cfg):
        self.ffn = FeedForward(**ffn_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x, attn_mask=None):

        def _inner_forward(x):
            x = self.attn(self.norm1(x), attn_mask=attn_mask)
            x = self.ffn(self.norm2(x))
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

@HEADS.register_module()
class GroupMaskTransformerHead(BaseDecodeHead):
    """Segmenter: Transformer for Semantic Segmentation.

    This head is the implementation of
    `Segmenter:　<https://arxiv.org/abs/2105.05633>`_.

    Args:
        backbone_cfg:(dict): Config of backbone of
            Context Path.
        in_channels (int): The number of channels of input image.
        num_layers (int): The depth of transformer.
        num_heads (int): The number of attention heads.
        embed_dims (int): The number of embedding dimension.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        drop_path_rate (float): stochastic depth rate. Default 0.1.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        init_std (float): The value of std in weight initialization.
            Default: 0.02.
    """

    def __init__(
            self,
            in_channels,
            in_index,
            channels,
            num_convs,
            num_classes,
            num_layers,
            num_heads,
            embed_dims,
            loss_decode,
            mlp_ratio=4,
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            num_fcs=2,
            qkv_bias=True,
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN'),
            init_std=0.02,
            **kwargs,
    ):
        super(GroupMaskTransformerHead, self).__init__(
            in_channels=in_channels, in_index=in_index, channels=channels, num_classes=num_classes, input_transform='resize_concat', loss_decode=loss_decode)
        channels_in = sum(in_channels)
        if isinstance(loss_decode, dict):
            self.loss_aux_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_aux_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_aux_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        self.conv_reduce = ConvModule(
                channels_in,
                channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                **kwargs
            )
        fcns_pre = []
        for i in range(num_convs):
            fcns_pre.append(ConvModule(
                channels,
                channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                **kwargs
            )) 
        self.add_module('fcns_pre', nn.Sequential(*fcns_pre))
        self.conv_seg_pre = nn.Conv2d(channels, self.num_classes, kernel_size=1)

#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 4, p2 = 4),
#             nn.Linear(16*embed_dims, embed_dims),
#         )
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerMaskEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    batch_first=True,
                ))

        self.dec_proj = nn.Linear(channels, embed_dims)
        self.decoder_norm = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)[1]
        self.patch_proj = nn.Linear(embed_dims, channels, bias=False)

        fcns = []
        for i in range(num_convs):
            fcns.append(ConvModule(
                channels,
                channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                **kwargs
            )) 
        self.add_module('fcns', nn.Sequential(*fcns))
               
        self.init_std = init_std

    def init_weights(self):
        trunc_normal_init(self.patch_proj, std=self.init_std)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=self.init_std, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x = self.conv_reduce(x)
        x_b = x
        x = self.fcns_pre(x)
        seg_pred = self.conv_seg_pre(x)
        
        seg_mask = seg_pred.detach().softmax(dim=1)
        b, _, h, w = seg_mask.shape
        seg_mask = resize(seg_mask, size=(h//4,w//4), mode='bilinear', align_corners=self.align_corners, warning=False)
        seg_mask = torch.argmax(seg_mask, dim=1) # N, H, W
        
        # print(torch.unique())
        seg_mask_ch = seg_mask.view(b, -1).unsqueeze(1).expand(-1, 2, -1)
        L = seg_mask_ch.shape[-1]
        seg_mask_group1 = torch.logical_not((seg_mask_ch < self.num_classes//3).bool())
        seg_mask_group2 = torch.logical_not(((seg_mask_ch < self.num_classes*2//3) & (seg_mask_ch >= self.num_classes//3)).bool())
        seg_mask_group3 = torch.logical_not(((seg_mask_ch < self.num_classes) & (seg_mask_ch >= self.num_classes*2//3)).bool())
        bz, hz, cz = seg_mask_group2.size()
        seg_mask_group4 = torch.zeros((bz, hz, cz), dtype=seg_mask_group2.dtype, device=seg_mask_group2.device)
        seg_attn_mask = torch.cat([seg_mask_group1, seg_mask_group2, seg_mask_group3, seg_mask_group4], dim=1)
        
        b, c, h, w = x_b.shape
        x_b = resize(x_b, size=(h//4,w//4), mode='bilinear', align_corners=self.align_corners, warning=False)
        x = x_b.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        x_ = self.dec_proj(x)
        # x = self.to_patch_embedding(x_b)
        
        for layer in self.layers:
            x = layer(x, attn_mask=seg_attn_mask)
        x = self.decoder_norm(x)
                                    
        proj_features = self.patch_proj(x).reshape(b, h//4, w//4, c).permute(0, 3, 1, 2)
        feats = self.fcns(proj_features)
        masks = self.cls_seg(feats)
        
        return seg_pred, masks

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
        seg_pred, seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, seg_pred, gt_semantic_seg)
        return losses
        
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
        _, seg_logits = self.forward(inputs)
        return seg_logits

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_logit1, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_logit1 = resize(
            input=seg_logit1,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
            seg_weight1 = self.sampler.sample(seg_logit1, seg_label)
        else:
            seg_weight = None
            seg_weight1 = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index) + loss_decode(
                    seg_logit1,
                    seg_label,
                    weight=seg_weight1,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += (loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index) + loss_decode(
                    seg_logit1,
                    seg_label,
                    weight=seg_weight1,
                    ignore_index=self.ignore_index))

        loss['acc_seg'] = accuracy(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss

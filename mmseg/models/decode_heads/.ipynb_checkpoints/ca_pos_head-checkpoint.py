# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from timm.models.layers import DropPath, trunc_normal_

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead

class Mlp(nn.Module):
    "Implementation of MLP"

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class RegionAttention(nn.Module):
    """
    Class attention layer from CaiT, see details in CaiT
    Class attention is the post stage in our VOLO, which is optional.
    """
    def __init__(self, dim, num_tokens=21, num_heads=8, head_dim=None, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            head_dim = dim // num_heads
            self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5

        self.kv = nn.Linear(dim,
                            self.head_dim * self.num_heads * 2,
                            bias=qkv_bias)
        self.q = nn.Linear(dim, self.head_dim * self.num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        kv = self.kv(x).reshape(B, N, 2, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)  # 2, B, self.num_heads, N, dim
        k, v = kv[0], kv[
            1]  # make torchscript happy (cannot use tensor as tuple)
        q = self.q(x[:, :self.num_tokens, :]).reshape(B, self.num_heads, self.num_tokens, self.head_dim) 
        attn = ((q * self.scale) @ k.transpose(-2, -1))  # B, self.num_heads, 21, N
        attn = attn.softmax(dim=-1)
        cls_attn = attn
        attn = self.attn_drop(attn)

        cls_embed = (attn @ v).transpose(1, 2).reshape( # B, self.num_heads, 21, dim
            B, self.num_tokens, self.head_dim * self.num_heads)
        cls_embed = self.proj(cls_embed)
        cls_embed = self.proj_drop(cls_embed)
        return cls_embed, cls_attn


class RegionBlock(nn.Module):
    """
    Class attention block from CaiT, see details in CaiT
    We use two-layers class attention in our VOLO, which is optional.
    """

    def __init__(self, dim, num_tokens, num_heads, head_dim=None, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_tokens = num_tokens
        self.norm1 = norm_layer(dim)
        self.attn = RegionAttention(
            dim, num_tokens=num_tokens, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.attn1 = RegionAttention(
            dim, num_tokens=num_tokens, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm3 = norm_layer(dim)

    def forward(self, x):
        cls_embed = x[:, :self.num_tokens]
        cls_embed1, _ = self.attn(self.norm1(x))
        cls_embed = cls_embed + self.drop_path(cls_embed1)
        cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))   # B, 21, num_heads, C
        x1 = torch.cat([cls_embed, x[:, self.num_tokens:]], dim=1)
        _, cls_attn = self.attn1(self.norm3(x1))
        return cls_attn



@HEADS.register_module()
class CAPosHead(BaseDecodeHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, num_tokens, num_heads, **kwargs):
        super(CAPosHead, self).__init__(**kwargs)
        self.num_tokens = num_tokens
        self.ra_block = RegionBlock(self.channels, num_tokens, num_heads)
        self.cls_token = nn.Parameter(torch.zeros(1, num_tokens, self.channels))
        trunc_normal_(self.cls_token, std=.02)
        
        self.conv_r = ConvModule(
                        self.in_channels, self.channels, 3, padding=1, 
                        conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
                    )
        self.conv_seg = nn.Conv2d(self.channels*num_tokens, self.out_channels, kernel_size=1)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 128*128+num_tokens, self.channels))
        
    def forward(self, inputs):
        """Forward function."""
        
        x = self._transform_inputs(inputs)
        output = self.conv_r(x)
        B, C, H, W = output.shape
        
        output1 = output.flatten(2, 3).permute(0, 2, 1)
        cls_token = self.cls_token.expand(output1.shape[0], -1, -1)
        output1 = torch.cat([cls_token, output1], dim=1) + self.pos_embed # B, N+21, C
        attn = self.ra_block(output1).mean(1)[:, :, self.num_tokens:].view(B, self.num_tokens, H, W).unsqueeze(2)  # B, N, C
        output = output.unsqueeze(1) * attn
        
        output = self.cls_seg(output.flatten(1, 2))
        return output

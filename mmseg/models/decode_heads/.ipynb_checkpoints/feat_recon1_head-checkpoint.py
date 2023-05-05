import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmseg.models.backbones.vit import TransformerEncoderLayer
from ..losses import accuracy

@HEADS.register_module()
class FeatRecon1Head(BaseDecodeHead):
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

        self.masked_conv = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels//2,
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg)
        self.decoder = TransformerEncoderLayer(self.channels//2, 8, self.channels//2)
        self.mask_token = nn.Parameter(torch.zeros(self.channels//2))
        torch.nn.init.normal_(self.mask_token, std=.02)

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
        seg, outs_masked, outs, mask = self(inputs)
        losses = self.losses(seg, gt_semantic_seg, outs_masked, outs, mask)
        return losses

    def forward(self, inputs):
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
        seg = self.conv_seg(out) 

        if self.training:
            out1 = self.masked_conv(out)
            B, C, H, W = out1.shape 
            out1 = resize(input=out1, size=(H//2, W//2), mode=self.interpolate_mode, align_corners=self.align_corners)
            out1 = out1.flatten(2, 3).permute(0, 2, 1)
            out_masked, mask = self.random_masking(out1)
            out_masked = self.decoder(out_masked)

            return seg, out_masked, out1.detach(), mask

        else:
            return seg 

    def random_masking(self, x, mask_ratio=0.75):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        # ids_keep = ids_shuffle[:, :len_keep]
        # x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        x_masked = x
        x_masked[mask == 1] = self.mask_token
        
        return x_masked, mask


    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, out_masked, out, mask):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                if loss_decode.loss_name == 'loss_dist':
                    loss[loss_decode.loss_name] = loss_decode(
                        out_masked * 10,
                        out * 10,
                        mask)
                else:
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logit,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
        
        # loss['recons'] = self.compute_l2_loss(out_masked, out, mask)
        loss['acc_seg'] = accuracy(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss

#     def compute_l2_loss(self, out_masked, out, mask):
#         loss = (out_masked - out) ** 2
#         loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
#         loss = (loss * mask).sum() / mask.sum()

#         return loss
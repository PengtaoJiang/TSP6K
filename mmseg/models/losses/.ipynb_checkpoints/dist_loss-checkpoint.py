# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

@LOSSES.register_module()
class DistLoss(nn.Module):

    def __init__(self, loss_type='L2'):
        super(DistLoss, self).__init__()
        self._loss_name = 'loss_dist'
        self.loss_type = loss_type
        
    def forward(self,
                pred,
                target,
                mask):
        assert target.requires_grad == False
        if self.loss_type == 'L1':
            loss = torch.abs(pred - target)
        else:
            loss = (pred - target) ** 2

        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum()
        #loss = (loss * mask).sum() / mask.sum()
        
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

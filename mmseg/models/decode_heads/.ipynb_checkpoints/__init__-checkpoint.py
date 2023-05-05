# Copyright (c) OpenMMLab. All rights reserved.
from .ann_head import ANNHead
from .apc_head import APCHead
from .aspp_head import ASPPHead
from .cc_head import CCHead
from .da_head import DAHead
from .dm_head import DMHead
from .dnl_head import DNLHead
from .dpt_head import DPTHead
from .ema_head import EMAHead
from .enc_head import EncHead
from .fcn_head import FCNHead
from .fpn_head import FPNHead
from .gc_head import GCHead
from .isa_head import ISAHead
from .knet_head import IterativeDecodeHead, KernelUpdateHead, KernelUpdator
from .lraspp_head import LRASPPHead
from .nl_head import NLHead
from .ocr_head import OCRHead
from .point_head import PointHead
from .psa_head import PSAHead
from .psp_head import PSPHead
from .psp_permute_head import PSPPermuteHead
from .segformer_head import SegformerHead
from .segmenter_mask_head import SegmenterMaskTransformerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .sep_aspp_swin_head import DepthwiseSeparableASPPSwinHead
from .sep_aspp_swin_ca_head import DepthwiseSeparableASPPSwinCAHead
from .sep_aspp_swin_ca4_head import DepthwiseSeparableASPPSwinCA4Head
from .sep_aspp_ca_head import DepthwiseSeparableASPPCAHead
from .sep_aspp_ca1_head import DepthwiseSeparableASPPCA1Head
from .sep_aspp_ca2_head import DepthwiseSeparableASPPCA2Head
from .sep_aspp_ca3_head import DepthwiseSeparableASPPCA3Head
from .sep_aspp_ca4_head import DepthwiseSeparableASPPCA4Head
from .sep_aspp_ra_head import DepthwiseSeparableASPPRAHead
from .sep_aspp_ra_mod_class_head import DepthwiseSeparableASPPRAClassHead
from .sep_aspp_ra1_head import DepthwiseSeparableASPPRA1Head
from .sep_aspp_ra_mod_head import DepthwiseSeparableASPPRAModHead
from .sep_aspp_ra_mod1_head import DepthwiseSeparableASPPRAMod1Head
from .sep_aspp_rares_head import DepthwiseSeparableASPPRAResHead
from .sep_aspp_rasum_head import DepthwiseSeparableASPPRASumHead
from .sep_aspp_rasum_res_head import DepthwiseSeparableASPPRASumResHead
from .sep_fcn_head import DepthwiseSeparableFCNHead
from .setr_mla_head import SETRMLAHead
from .setr_up_head import SETRUPHead
from .stdc_head import STDCHead
from .uper_head import UPerHead
from .uper_ca_head import UPerCAHead
from .uper_ca3_head import UPerCA3Head
from .uper_swin_head import UPerSwinHead
from .ham_head import LightHamHead
from .group_mask_head import GroupMaskTransformerHead
from .nl_enc_head import NLEncHead
from .ca_head import CAHead
from .ra_head import RAHead
from .ca_pos_head import CAPosHead
from .ca_res_head import CAResHead
from .ca_cc_head import CACCHead
from .casum_res_head import CASumResHead
from .casum_head import CASumHead

from .segformer_diffusion_head import SegformerDiffusionHead
from .feat_recon_head import FeatReconHead
from .seg_recon_head import SegReconHead
from .seg_recon1_head import SegRecon1Head
from .feat_recon1_head import FeatRecon1Head
from .fm_head import FMHead

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'PSAHead', 'NLHead', 'GCHead', 'CCHead',
    'UPerHead', 'DepthwiseSeparableASPPHead', 'ANNHead', 'DAHead', 'OCRHead',
    'EncHead', 'DepthwiseSeparableFCNHead', 'FPNHead', 'EMAHead', 'DNLHead',
    'PointHead', 'APCHead', 'DMHead', 'LRASPPHead', 'SETRUPHead',
    'SETRMLAHead', 'DPTHead', 'SETRMLAHead', 'SegmenterMaskTransformerHead',
    'SegformerHead', 'ISAHead', 'STDCHead', 'IterativeDecodeHead',
    'KernelUpdateHead', 'KernelUpdator', 'LightHamHead', 'PSPPermuteHead', 
    'GroupMaskTransformerHead', 'DepthwiseSeparableASPPSwinHead', 'NLEncHead',
    'DepthwiseSeparableASPPCAHead', 'DepthwiseSeparableASPPCA1Head',
    'DepthwiseSeparableASPPCA2Head', 'DepthwiseSeparableASPPCA3Head', 
    'UPerSwinHead', 'UPerCAHead', 'UPerCA3Head', 'DepthwiseSeparableASPPSwinCAHead',
    'DepthwiseSeparableASPPCA4Head', 'DepthwiseSeparableASPPSwinCA4Head', 
    'DepthwiseSeparableASPPRAHead', 'CAHead', 'CAResHead', 
    'DepthwiseSeparableASPPRAResHead', 'CAPosHead', 'CACCHead', 'DepthwiseSeparableASPPRA1Head',
    'DepthwiseSeparableASPPRASumHead', 'CASumResHead', 'CASumHead', 'DepthwiseSeparableASPPRASumResHead',
    'SegformerDiffusionHead', 'FeatRecon1Head', 'SegReconHead', 'SegRecon1Head', 'FMHead', 
    'DepthwiseSeparableASPPRAMod1Head', 'DepthwiseSeparableASPPRAClassHead', 'RAHead'
]

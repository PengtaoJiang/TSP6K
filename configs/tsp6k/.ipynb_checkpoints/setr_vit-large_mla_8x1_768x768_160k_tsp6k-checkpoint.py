_base_ = [
    '../_base_/models/setr_mla.py', '../_base_/datasets/tsp6k_768x768.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained=None,
    backbone=dict(
        drop_rate=0,
        init_cfg=dict(
            type='Pretrained', checkpoint='./pretrained/vit_large_p16.pth')),
    decode_head=dict(
        type='SETRMLAHead',
        in_channels=(256, 256, 256, 256),
        channels=512,
        in_index=(0, 1, 2, 3),
        dropout_ratio=0,
        mla_channels=128,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=0,
            dropout_ratio=0,
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=21,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=1,
            dropout_ratio=0,
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=21,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=2,
            dropout_ratio=0,
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=21,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=3,
            dropout_ratio=0,
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=21,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    ],
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512)))

optimizer = dict(
    lr=0.002,
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))
data = dict(samples_per_gpu=1)




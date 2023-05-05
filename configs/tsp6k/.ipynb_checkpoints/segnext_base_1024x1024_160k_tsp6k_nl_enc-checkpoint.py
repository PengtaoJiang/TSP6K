_base_ = [
    '../_base_/datasets/tsp6k_1024x1024.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]


# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MSCAN',
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.2,
        depths=[3, 3, 12, 3],
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/mscan_b.pth'),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    decode_head=dict(
        type='NLEncHead',
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        input_transform='resize_concat',
        channels=512,
        c1_in_channels=128, 
        c1_channels=48,
        dropout_ratio=0.1,
        reduction=2,
        use_scale=True,
        mode='embedded_gaussian',
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=320,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'))
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

# data
data = dict(samples_per_gpu=1)
evaluation = dict(interval=16000, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=16000)
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

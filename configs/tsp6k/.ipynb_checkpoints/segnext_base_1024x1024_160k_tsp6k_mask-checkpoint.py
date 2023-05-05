_base_ = [
    '../_base_/datasets/tsp6k_1024x1024.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]


# model settings
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
        type='GroupMaskTransformerHead',
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=384,
        num_convs=2,
        num_classes=21,
        num_layers=2,
        num_heads=8,
        embed_dims=384,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'))
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

# data
data = dict(samples_per_gpu=2)
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

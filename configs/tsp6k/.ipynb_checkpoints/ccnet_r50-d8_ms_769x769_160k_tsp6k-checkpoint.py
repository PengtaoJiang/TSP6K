_base_ = [
    '../_base_/models/ccnet_r50-d8.py',
    '../_base_/datasets/tsp6k_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    decode_head=dict(
        type='CCHead',
        in_channels=[512, 1024, 2048],
        in_index=[1, 2, 3],
        input_transform='resize_concat',
        channels=512,
        recurrence=2,
        dropout_ratio=0.1,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(align_corners=True, num_classes=21),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))

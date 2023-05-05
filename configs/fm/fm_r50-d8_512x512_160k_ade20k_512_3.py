_base_ = [
    '../_base_/models/fm_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    decode_head=dict(fm_channels=512, window_size=3, num_classes=150), auxiliary_head=dict(num_classes=150))


# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# crop_size = (240, 240)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', reduce_zero_label=True),
#     dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg']),
# ]
# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=1,
#     train=dict(
#         pipeline=train_pipeline))
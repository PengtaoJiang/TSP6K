_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/tsp6k_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
model = dict(
    decode_head=dict(align_corners=True, num_classes=21),
    auxiliary_head=dict(align_corners=True, num_classes=21),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))


# data = dict(samples_per_gpu=1, workers_per_gpu=2)
# crop_size = (120, 120)
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),
#     dict(type='Resize', img_scale=(2049, 1025), ratio_range=(0.5, 2.0)),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg']),
# ]

# data = dict(
#     train=dict(pipeline=train_pipeline))
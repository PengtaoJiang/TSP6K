_base_ = [
    '../_base_/models/ccnet_r50-d8.py',
    '../_base_/datasets/tsp6k_1024x1024.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
model = dict(
    decode_head=dict(align_corners=True, num_classes=21),
    auxiliary_head=dict(align_corners=True, num_classes=21),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

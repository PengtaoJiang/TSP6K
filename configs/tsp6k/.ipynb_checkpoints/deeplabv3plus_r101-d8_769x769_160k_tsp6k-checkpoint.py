_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/tsp6k_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

model = dict(
    pretrained='./pretrained/resnet101_v1c-e67eebb6.pth', 
    backbone=dict(depth=101),
    decode_head=dict(align_corners=True, num_classes=21),
    auxiliary_head=dict(align_corners=True, num_classes=21),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))

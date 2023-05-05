_base_ = ['./segformer_mit-b0_8x1_1024x1024_160k_tsp6k.py']

checkpoint = './pretrained/mit_b2_20220624-66e8bf70.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=21))

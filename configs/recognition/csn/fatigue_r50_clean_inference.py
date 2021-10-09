_base_ = [
    '../../_base_/models/ircsn_r152.py', '../../_base_/default_runtime.py'
]

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dCSN',
        pretrained2d=False,
        pretrained=None,
        depth=50,
        with_pool2=False,
        bottleneck_mode='ir',
        norm_eval=True,
        bn_frozen=True,
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=2,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='org_prob'))

# dataset settings
clip_len = 48
dataset_type = 'VideoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
test_pipeline = [
    dict(type='OpenCVInit', num_threads=1),
    dict(
        type='SampleFrames',
        clip_len=clip_len,
        frame_interval=1,
        num_clips=10,
        test_mode=True),
    dict(type='OpenCVDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=1,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        ann_file=None,
        data_prefix=None,
        pipeline=test_pipeline))


_base_ = ['../../_base_/default_runtime.py']

# model settings
in_channels = 1
base_channels = 16
max_channels = 128
stem_stride = 2

clip_len = 32

hidden_size = 64
layers_num = 1
num_segments = clip_len

model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetTiny',
        pretrained=None,
        torchvision_pretrain=False,
        depth=14,
        in_channels = in_channels,
        base_channels = base_channels,
        max_channels = max_channels,
        stem_stride = stem_stride),
    cls_head=dict(
        type='LSTMHead',
        num_classes=2,
        in_channels=128,
        num_segments=num_segments,
        hidden_size=hidden_size,
        layers_num=layers_num,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.001),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# dataset settings
dataset_type = 'FatigueCleanDatasetWithoutDahaqian'
data_root = '/zhourui/workspace/pro/fatigue/data/rawframes/new_clean'
data_root_val = '/zhourui/workspace/pro/fatigue/data/rawframes/new_clean'
facerect_data_prefix = '/zhourui/workspace/pro/fatigue/data/anns/new_clean'
ann_file_train = '/zhourui/workspace/pro/fatigue/data/anns/new_clean/20211108_fatigue_lookdown_squint_calling_smoking_dahaqian.json'
ann_file_val = '/zhourui/workspace/pro/fatigue/data/anns/new_clean/20211108_fatigue_lookdown_squint_calling_smoking_dahaqian.json'
ann_file_test = '/zhourui/workspace/pro/fatigue/data/anns/new_clean/20211108_fatigue_lookdown_squint_calling_smoking_dahaqian.json'
test_save_results_path = 'work_dirs/cnn_lstm/valid_results_testone.npy'
test_save_label_path = 'work_dirs/cnn_lstm/valid_label_testone.npy'

img_norm_cfg = dict(
    mean=[127.5], std=[127.5], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=1, num_clips=1, out_of_bound_opt='repeat_last'),
    dict(type='FatigueRawFrameDecodeGray'),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(112, 112), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='NormalizeGray', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=clip_len,
        frame_interval=1,
        num_clips=1,
        test_mode=True,
        out_of_bound_opt='repeat_last'),
    dict(type='FatigueRawFrameDecodeGray'),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='NormalizeGray', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=clip_len,
        frame_interval=1,
        num_clips=1,
        test_mode=True,
        out_of_bound_opt='repeat_last'),
    dict(type='FatigueRawFrameDecodeGray'),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='ThreeCrop', crop_size=128),
    dict(type='NormalizeGray', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=6,
    pin_memory=False,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        video_data_prefix=data_root,
        facerect_data_prefix=facerect_data_prefix,
        data_phase='train',
        test_mode=False,
        pipeline=train_pipeline,
        min_frames_before_fatigue=clip_len),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        video_data_prefix=data_root_val,
        facerect_data_prefix=facerect_data_prefix,
        data_phase='valid',
        test_mode=True,
        test_all=False,
        pipeline=val_pipeline,
        min_frames_before_fatigue=clip_len),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        video_data_prefix=data_root_val,
        facerect_data_prefix=facerect_data_prefix,
        data_phase='valid',
        test_mode=True,
        test_all=False,
        test_save_label_path=test_save_label_path,
        test_save_results_path=test_save_results_path,
        pipeline=test_pipeline,
        min_frames_before_fatigue=clip_len))
evaluation = dict(
    interval=5, metrics=['top_k_classes'])

# optimizer
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[10, 25, 48],
    warmup='linear',
    warmup_ratio=0.1,
    warmup_by_epoch=True,
    warmup_iters=16)
total_epochs = 58

work_dir = './work_dirs/cnn_lstm'  # noqa: E501
find_unused_parameters = True

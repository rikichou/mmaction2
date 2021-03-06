_base_ = [
    '../../_base_/schedules/sgd_tsm_mobilenet_v2_100e.py',
    '../../_base_/default_runtime.py'
]
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='MobileNetV2TSM',
        shift_div=8,
        num_segments=8,
        is_shift=True,
        pretrained='mmcls://mobilenet_v2'),
    cls_head=dict(
        type='TSMHead',
        num_segments=8,
        num_classes=2,
        in_channels=1280,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

dataset_type = 'FatigueCleanDataset'
data_root = '/zhourui/workspace/pro/fatigue/data/rawframes/new_clean'
data_root_val = '/zhourui/workspace/pro/fatigue/data/rawframes/new_clean'
facerect_data_prefix = '/zhourui/workspace/pro/fatigue/data/anns/new_clean'
ann_file_train = '/zhourui/workspace/pro/fatigue/data/anns/new_clean/20211108_fatigue_lookdown_squint_calling_smoking_dahaqian.json'
ann_file_val = '/zhourui/workspace/pro/fatigue/data/anns/new_clean/20211108_fatigue_lookdown_squint_calling_smoking_dahaqian.json'
ann_file_test = '/zhourui/workspace/pro/fatigue/data/anns/new_clean/20211108_fatigue_lookdown_squint_calling_smoking_dahaqian.json'
test_save_results_path = 'work_dirs/fatigue_r50_clean_with_squint_smoke_call_dahaqian/valid_results_testone.npy'
test_save_label_path = 'work_dirs/fatigue_r50_clean_with_squint_smoke_call_dahaqian/valid_label_testone.npy'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
clip_len = 8
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=clip_len, out_of_bound_opt='repeat_last'),
    dict(type='FatigueRawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=clip_len,
        test_mode=True,
        out_of_bound_opt='repeat_last'),
    dict(type='FatigueRawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=clip_len,
        test_mode=True,
        out_of_bound_opt='repeat_last'),
    dict(type='FatigueRawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=4,
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
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001)

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/debug/'

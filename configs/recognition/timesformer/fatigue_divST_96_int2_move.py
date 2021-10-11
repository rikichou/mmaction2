_base_ = ['../../_base_/default_runtime.py']

# model settings
clip_len = 48
frame_interval = 2
min_frames_before_fatigue = 96 # 48 * 2
input_size = 112
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='TimeSformer',
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/recognition/timesformer/vit_base_patch16_224.pth',  # noqa: E501
        num_frames=clip_len,
        img_size=input_size,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.,
        transformer_layers=None,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(type='TimeSformerHead', num_classes=2, in_channels=768),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# dataset settings
dataset_type = 'FatigueMoveDataset'
data_root = '/zhourui/workspace/pro/fatigue/data/rawframes/new_clean/fatigue_clips'
data_root_val = '/zhourui/workspace/pro/fatigue/data/rawframes/new_clean/fatigue_clips'
facerect_data_prefix = '/zhourui/workspace/pro/fatigue/data/clean/fatigue_info_from_yolov5'
ann_file_train = '/zhourui/workspace/pro/fatigue/data/clean/fatigue_anns/20210824_fatigue_pl_less_than_50_fatigue_full_info_all_path.json'
ann_file_val = '/zhourui/workspace/pro/fatigue/data/clean/fatigue_anns/20210824_fatigue_pl_less_than_50_fatigue_full_info_all_path.json'
ann_file_test = '/zhourui/workspace/pro/fatigue/data/clean/fatigue_anns/20210824_fatigue_pl_less_than_50_fatigue_full_info_all_path.json'
test_save_results_path = 'work_dirs/fatigue_timesformer_divST_96_int2_move/valid_results_testone.npy'
test_save_label_path = 'work_dirs/fatigue_timesformer_divST_96_int2_move/valid_label_testone.npy'

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=1, out_of_bound_opt='repeat_last'),
    dict(type='FatigueRawFrameDecode'),
    dict(type='RandomRescale', scale_range=(128, 160)),
    dict(type='RandomCrop', size=input_size),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
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
    dict(type='FatigueRawFrameDecode'),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='CenterCrop', crop_size=input_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=clip_len,
        frame_interval=1,
        num_clips=1,
        test_mode=True,
        out_of_bound_opt='repeat_last'),
    dict(type='FatigueRawFrameDecode'),
    dict(type='Resize', scale=(-1, 112)),
    dict(type='ThreeCrop', crop_size=112),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=6,
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
        min_frames_before_fatigue=min_frames_before_fatigue),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        video_data_prefix=data_root_val,
        facerect_data_prefix=facerect_data_prefix,
        data_phase='valid',
        test_mode=True,
        test_all=False,
        pipeline=val_pipeline,
        min_frames_before_fatigue=min_frames_before_fatigue),
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
        min_frames_before_fatigue=min_frames_before_fatigue))

evaluation = dict(
    interval=5, metrics=['top_k_classes'])

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        }),
    weight_decay=1e-4,
    nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[5, 10, 15, 20])
total_epochs = 30

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = './work_dirs/fatigue_timesformer_divST_96_int2_move'

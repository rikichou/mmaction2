_base_ = [
    '../../_base_/models/ircsn_r152.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(
        depth=50,
        norm_eval=True,
        bn_frozen=True,
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/recognition/csn/ircsn_from_scratch_r50_ig65m_20210617-ce545a37.pth'  # noqa: E501
    ),
    cls_head=dict(
        num_classes=2
    )
)

# dataset settings
dataset_type = 'FatigueRawframeDataset'
data_root = 'data/fatigue/train'
data_root_val = 'data/fatigue/valid'
ann_file_train = 'data/fatigue/train/streamax_train_list_rawframes.txt'
ann_file_val = 'data/fatigue/valid/streamax_val_list_rawframes.txt'
ann_file_test = 'data/fatigue/valid/streamax_val_list_rawframes.txt'
test_save_results_path = 'work_dirs/fatigue_r50_randomchoose/valid_results_testone.npy'
test_save_label_path = 'work_dirs/fatigue_r50_randomchoose/valid_label_testone.npy'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
clip_len = 48
train_pipeline = [
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=1, num_clips=1, out_of_bound_opt='repeat_last'),
    dict(type='FatigueRawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
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
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
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
    dict(type='FatigueRawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=6,
    workers_per_gpu=4,
    pin_memory=False,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        min_frames_before_fatigue=clip_len),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        min_frames_before_fatigue=clip_len),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        min_frames_before_fatigue=clip_len,
        test_all=False,
        test_save_results_path=test_save_results_path,
        test_save_label_path=test_save_label_path
        ))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD', lr=0.000125, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[32, 48],
    warmup='linear',
    warmup_ratio=0.1,
    warmup_by_epoch=True,
    warmup_iters=16)
total_epochs = 58

work_dir = './work_dirs/fatigue_r50_randomchoose'  # noqa: E501
find_unused_parameters = True

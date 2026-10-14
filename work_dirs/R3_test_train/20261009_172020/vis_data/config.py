crop_size = (
    512,
    512,
)
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mmseg_custom.openearthmap',
    ])
data_root = ''
dataset_type = 'OpenEarthMapDataset'
default_scope = 'mmseg'
model = dict(
    backbone=dict(
        embed_dims=32,
        num_heads=[
            1,
            2,
            5,
            8,
        ],
        num_layers=[
            2,
            2,
            2,
            2,
        ],
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        sr_ratios=[
            8,
            4,
            2,
            1,
        ],
        type='MixVisionTransformer'),
    decode_head=dict(
        channels=256,
        in_channels=[
            32,
            64,
            160,
            256,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        num_classes=9,
        type='SegformerHead'),
    type='EncoderDecoder')
optim_wrapper = dict(optimizer=dict(lr=6e-05, type='AdamW', weight_decay=0.01))
test_cfg = None
test_dataloader = None
test_evaluator = None
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=10, type='IterBasedTrainLoop', val_interval=10)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='D:\\SegFormer_Project\\work_dirs\\R0_audit\\train_pool.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        reduce_zero_label=False,
        type='OpenEarthMapDataset'),
    num_workers=0)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
]
val_cfg = None
val_dataloader = None
val_evaluator = None
work_dir = 'D:\\SegFormer_Project\\work_dirs\\R3_test_train'

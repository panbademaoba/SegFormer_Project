custom_imports = dict(
    imports=[
        'mmseg_custom.openearthmap',
    ],
    allow_failed_imports=False
)


dataset_type = 'OpenEarthMapDataset'


data_root = ''


crop_size = (512, 512)



train_pipeline = [

    dict(
        type='LoadImageFromFile'
    ),


    dict(
        type='LoadAnnotations'
    ),


    dict(
        type='Resize',
        scale=(512, 512)
    ),


    dict(
        type='PackSegInputs'
    )
]



test_pipeline = [

    dict(
        type='LoadImageFromFile'
    ),


    dict(
        type='LoadAnnotations'
    ),


    dict(
        type='Resize',
        scale=(512, 512)
    ),


    dict(
        type='PackSegInputs'
    )
]



train_dataloader = dict(

    batch_size=2,

    num_workers=0,


    dataset=dict(

        type=dataset_type,


        ann_file=
        r'D:\SegFormer_Project\work_dirs\R0_audit\train_pool.txt',


        reduce_zero_label=False,


        pipeline=train_pipeline
    )
)



val_dataloader = dict(

    batch_size=1,

    num_workers=0,


    dataset=dict(

        type=dataset_type,


        ann_file=
        r'D:\SegFormer_Project\work_dirs\R0_audit\iid_val.txt',


        reduce_zero_label=False,


        pipeline=test_pipeline
    )
)




model = dict(

    type='EncoderDecoder',


    data_preprocessor=dict(

        type='SegDataPreProcessor',


        mean=[
            123.675,
            116.28,
            103.53
        ],


        std=[
            58.395,
            57.12,
            57.375
        ],


        bgr_to_rgb=True,


        pad_val=0,


        seg_pad_val=255,


        size=(512, 512)
    ),



    backbone=dict(

        type='MixVisionTransformer',


        embed_dims=32,


        num_stages=4,


        num_layers=[
            2,
            2,
            2,
            2
        ],


        num_heads=[
            1,
            2,
            5,
            8
        ],


        patch_sizes=[
            7,
            3,
            3,
            3
        ],


        sr_ratios=[
            8,
            4,
            2,
            1
        ],


        out_indices=(
            0,
            1,
            2,
            3
        )
    ),



    decode_head=dict(

        type='SegformerHead',


        in_channels=[
            32,
            64,
            160,
            256
        ],


        in_index=[
            0,
            1,
            2,
            3
        ],


        channels=256,


        num_classes=9
    )
)




train_cfg = dict(

    type='IterBasedTrainLoop',

    max_iters=10,

    val_interval=10
)



val_cfg = dict(
    type='ValLoop'
)



test_cfg = dict(
    type='TestLoop'
)



optim_wrapper = dict(

    optimizer=dict(

        type='AdamW',

        lr=0.00006,

        weight_decay=0.01
    )
)



default_scope = 'mmseg'



work_dir = './work_dirs/segformer_b0_openearthmap'
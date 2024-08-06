_base_ = [
    # 'mmdet::_base_/datasets/coco_detection.py',mmdetection/configs/_base_/datasets/mycocodet.py
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

data_root = '/home/gsw/mmdetection/data/coco/'

# classes = ("BS","CZ","KJ")
metainfo = {
    'classes':
    ( 'Blueberry leaf' , 'Tomato leaf yellow virus' , 'Peach leaf' , 'Raspberry leaf' , 
     'Strawberry leaf' , 'Tomato Septoria leaf spot' , 'Tomato leaf' , 'Corn leaf blight' , 
     'Potato leaf early blight' , 'Bell_pepper leaf' , 'Tomato mold leaf' , 'Tomato leaf bacterial spot' , 
     'Soyabean leaf' , 'Bell_pepper leaf spot' , 'Tomato leaf mosaic virus' , 'Squash Powdery mildew leaf' ,
       'Potato leaf late blight' , 'Apple leaf' , 'Cherry leaf' , 'Tomato leaf late blight' , 'grape leaf' , 
       'Tomato Early blight leaf' , 'Apple rust leaf' , 'Apple Scab Leaf' , 'grape leaf black rot' ,
         'Corn rust leaf' , 'Corn Gray leaf spot' , 'Potato leaf' , 'Tomato two spotted spider mites leaf'),
    # palette is a list of color tuples, which is used for visualization.
    'palette':
    [    (220, 20, 60), (119, 11, 32),   (0, 0, 142),   (0, 0, 230),
         (0, 60, 100), (0, 80, 100),    (0, 0, 70),      (0, 0, 192),
         (100, 170, 30),(220, 220, 0),  (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255),  (0, 226, 252),  (182, 182, 255),
         (0, 82, 0),    (120, 166, 157), (110, 76, 0),    (174, 57, 255),
         (199, 100, 0), (72, 0, 118),    (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176),  (255, 99, 164),
         (92, 0, 73)
         ]
    }
num_classes = 29

custom_imports = dict(
    imports=['projects.DiffusionDet.diffusiondet'], allow_failed_imports=False)

# model settings
model = dict(
    type='DiffusionDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    bbox_head=dict(
        type='DynamicDiffusionDetHead',
        num_classes=29,
        feat_channels=256,
        num_proposals=500,
        num_heads=6,
        deep_supervision=True,
        prior_prob=0.01,
        snr_scale=2.0,
        sampling_timesteps=1,
        ddim_sampling_eta=1.0,
        single_head=dict(
            type='SingleDiffusionDetHead',
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            dropout=0.0,
            act_cfg=dict(type='ReLU', inplace=True),
            dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)),
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        # criterion
        criterion=dict(
            type='DiffusionDetCriterion',
            num_classes=29,
            assigner=dict(
                type='DiffusionDetMatcher',
                match_costs=[
                    dict(
                        type='FocalLossCost',
                        alpha=0.25,
                        gamma=2.0,
                        weight=2.0,
                        eps=1e-8),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)
                ],
                center_radius=2.5,
                candidate_topk=5),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                alpha=0.25,
                gamma=2.0,
                reduction='sum',
                loss_weight=2.0),
            loss_bbox=dict(type='L1Loss', reduction='sum', loss_weight=5.0),
            loss_giou=dict(type='GIoULoss', reduction='sum',
                           loss_weight=2.0))),
    test_cfg=dict(
        use_nms=True,
        score_thr=0.5,
        min_bbox_size=0,
        nms=dict(type='nms', iou_threshold=0.5),
    ))

backend = 'pillow'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend=backend),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='RandomChoiceResize',
                scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True,
                backend=backend),
        ],
                    [
                        dict(
                            type='RandomChoiceResize',
                            scales=[(400, 1333), (500, 1333), (600, 1333)],
                            keep_ratio=True,
                            backend=backend),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='RandomChoiceResize',
                            scales=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            keep_ratio=True,
                            backend=backend)
                    ]]),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend=backend),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True, backend=backend),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
# train_dataloader = dict(
#     sampler=dict(type='InfiniteSampler'),
#     dataset=dict(
#         # metainfo=dict(classes=classes),
#         filter_cfg=dict(filter_empty_gt=False, min_size=1e-5),
#         pipeline=train_pipeline))
val_batch_size_per_gpu = 1
val_num_workers = 2
train_batch_size_per_gpu = 12
# 可以根据自己的电脑修改
train_num_workers = 4

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    sampler=dict(type='InfiniteSampler'),
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False, min_size=1e-5),
        pipeline=train_pipeline,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='images/')))

# val_dataloader = dict(dataset=dict(pipeline=test_pipeline,
#                                 #    metainfo=dict(classes=classes)
#                                    ))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        pipeline=test_pipeline,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True, type='AdamW', lr=0.000025, weight_decay=0.0001),
    clip_grad=dict(max_norm=1.0, norm_type=2))
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=45000,
    val_interval=7500)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=45000,
        by_epoch=False,
        milestones=[35000, 42000],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=7500, max_keep_ckpts=3))
log_processor = dict(by_epoch=False)

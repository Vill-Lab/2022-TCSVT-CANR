_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4.py',
    '../_base_/datasets/cuhk.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
norm_cfg = dict(type='BN', requires_grad=False)

model = dict(
    type="PersonSearchE2E",
    roi_head=dict(
        type='PersonSearchRoIHead',
        shared_head=dict(
            type='ResLayer',
            depth=50,
            stage=3,
            stride=2,
            dilation=1,
            style='caffe',
            norm_cfg=norm_cfg,
            norm_eval=True),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[16]),
        attention_policy=10,
        bbox_head=dict(
            type='PersonSearchBBoxHead',
            reid_dim=256,
            reid_bias=False,
            num_pid=5532,
            size_queue=5000,
            oim_momentum=0.5,
            oim_temp=30,
            gt_update_only=False,
            reid_max_pooling=True,
            vib=True,
            reid_ftbn=True,
            cls_reg_bn=True,
            ce="weighted2",
            weight_mat="lut_cuhk.pth",
            contrastive_loss=True,
            cont_loss_policy=4,
            cont_scale_ax=-5,
            oic_queue=0,
            oic_scale=30,
            arcface=None,
            with_avg_pool=True,
            roi_feat_size=7,
            in_channels=2048,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))))

# model training and testing settings
train_cfg = dict(
    rcnn=dict(
        sampler=dict(
            type="BalancedPosSamplerWithGT",
            pos_fraction=0.4,
            num=128,),
        debug=True))

optimizer = dict(type="SGD", lr=0.004, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=10, norm_type=2))

evaluation = dict(interval=100, gallery_size=100)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
)

lr_config = dict(
    warmup_iters=1400,
)

test_cfg = dict(
    rpn=dict(
        nms_post=300,
    ),
    rcnn=dict(
        nms=dict(type='nms', iou_threshold=0.4),
    )
)

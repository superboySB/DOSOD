_base_ = ('../../third_party/mmyolo/configs/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# hyper-parameters
num_classes = 80
num_training_classes = 80
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]

# model settings
model = dict(type='SimpleYOLOWorldDetector',
             mm_neck=True,
             num_train_classes=num_classes,
             num_test_classes=num_classes,
             reparameterized=True,
             data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
             backbone=dict(_delete_=True,
                           type='MultiModalYOLOBackbone',
                           text_model=None,
                           image_model={{_base_.model.backbone}},
                           with_text_model=False),
             neck=dict(type='YOLOWorldPAFPN',
                       guide_channels=num_classes,
                       embed_channels=neck_embed_channels,
                       num_heads=neck_num_heads,
                       block_cfg=dict(type='RepConvMaxSigmoidCSPLayerWithTwoConv',
                                      guide_channels=num_classes)),
             bbox_head=dict(head_module=dict(type='RepYOLOWorldHeadModule',
                                             embed_dims=text_channels,
                                             num_guide=num_classes,
                                             num_classes=num_classes)),
             train_cfg=dict(assigner=dict(num_classes=num_classes)))

_base_ = '../../third_party/mmyolo/configs/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco.py'
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# hyper-parameters
num_training_classes = 80  # lvis 1202, coco 80
text_channels = 512
joint_space_dims = 512

# model settings
model = dict(
    type='RepDOSODDetector',
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    bbox_head=dict(type='RepDOSODYOLOv8Head',
                   head_module=dict(type='RepDOSODYOLOv8dHeadModuleDRobotics',
                                    text_embed_dims=text_channels,
                                    joint_space_dims=joint_space_dims,
                                    num_classes=num_training_classes)))

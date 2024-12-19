import os
from ultralytics import YOLO

# Enable CUDA launch blocking for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Hyperparameters for training
hyp = {
    'lr0': 0.001,  # initial learning rate
    'lrf': 0.0002,  # final learning rate (lr0 * lrf)
    'momentum': 0.9,  # SGD momentum/Adam beta1
    'weight_decay': 0.0005,  # optimizer weight decay
    'warmup_epochs': 3.0,  # warmup epochs
    'warmup_momentum': 0.8,  # warmup initial momentum
    'warmup_bias_lr': 0.1,  # warmup initial bias lr
    'box': 7.5,  # box loss gain
    'cls': 0.5,  # cls loss gain (scale with pixels)
    'dfl': 1.5,  # dfl loss gain
    'pose': 12.0,  # pose loss gain
    'kobj': 1.0,  # keypoint obj loss gain
    'label_smoothing': 0.0,  # label smoothing (fraction)
    'nbs': 64,  # nominal batch size
    'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
    'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
    'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
    'degrees': 0.0,  # image rotation (+/- deg)
    'translate': 0.1,  # image translation (+/- fraction)
    'scale': 0.5,  # image scale (+/- gain)
    'shear': 0.0,  # image shear (+/- deg)
    'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
    'flipud': 0.5,  # image flip up-down (probability)
    'fliplr': 0.5,  # image flip left-right (probability)
    'mosaic': 1.0,  # image mosaic (probability)
    'mixup': 0.2,  # image mixup (probability)
    'copy_paste': 0.3  # segment copy-paste (probability)
}

# Load the model configuration and weights
model = YOLO('yolov8n.yaml').load('yolov8n.pt')

# Train the model with the provided hyperparameters and dataset
results = model.train(
    data='coco.yaml',  # path to your dataset configuration file
    epochs=50,  # number of epochs
    imgsz=640,  # image size
    batch=4,  # batch size
    workers=4,  # number of workers
    amp=False,  # automatic mixed precision
    val=True,  # enable validation after each epoch
    **hyp  # Unpack the hyperparameters
)
import os

project_root_dir= os.getcwd() + "/../"
data_dir= project_root_dir + "data/"
images_train_dir= data_dir + "images/train/"
images_val_dir= data_dir + "images/val/"
labels_train_dir= data_dir + "labels/train/"
labels_2_dir= data_dir + "labels2/"
labels_val_dir= data_dir + "labels/val/"
artifacts_dir= project_root_dir + "artifacts/"

FEATURE_MAP_SHAPE= (32, 32)
IMAGE_SIZE_RESHAPED= (512, 512)
ANCHOR_SCALES= [16, 32, 64, 128, 256]
ANCHOR_SCALES_2= [16, 32, 64, 128, 256]
ANCHOR_RATIOS= [0.5, 1, 2]
ANCHOR_RATIOS_2= [0.5, 0.75, 1, 1.5, 2]
NUM_ANCHORS_PER_LOC= len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
BATCH_SIZE= 16
LEARNING_RATE= 1e-4
NUM_EPOCHS= 5
IOU_HIGH_THRESHOLD= 0.7
IOU_LOW_THRESHOLD= 0.3
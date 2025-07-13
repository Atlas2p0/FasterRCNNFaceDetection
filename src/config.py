import os

project_root_dir= os.getcwd() + "/../"
data_dir= project_root_dir + "data/"
images_train_dir= data_dir + "images/train/"
images_val_dir= data_dir + "images/val/"
labels_train_dir= data_dir + "labels/train/"
labels_val_dir= data_dir + "labels/val/"
artifacts_dir= project_root_dir + "artifacts/"


FEATURE_MAP_SHAPE= (14, 14)
IMAGE_SIZE_RESHAPED= (224, 224)
ANCHOR_STRIDE= IMAGE_SIZE_RESHAPED[0] / FEATURE_MAP_SHAPE[0]
ANCHOR_SCALES= [8, 16, 24, 32, 64, 128]
ANCHOR_RATIOS= [0.5, 0.75, 1.0, 1.33, 2.0]
NUM_ANCHORS_PER_LOC= len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
BATCH_SIZE= 16
LEARNING_RATE= 1e-3
EPOCHS= 5
IOU_HIGH_THRESHOLD= 0.6
IOU_LOW_THRESHOLD= 0.3
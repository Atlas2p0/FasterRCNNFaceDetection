import os

project_root_dir= os.getcwd() + "/../"
data_dir= project_root_dir + "data/"
images_train_dir= data_dir + "images/train/"
images_val_dir= data_dir + "images/val/"
labels_train_dir= data_dir + "labels/train/"
labels_val_dir= data_dir + "labels/val/"
artifacts_dir= project_root_dir + "artifacts/"


FEATURE_MAP_SHAPE= (28, 28)
IMAGE_SIZE_RESHAPED= (224, 224)
ANCHOR_STRIDE= 8
ANCHOR_SCALES= [4, 16, 24, 32, 64]
ANCHOR_RATIOS= [0.5, 0.75, 1, 1.5, 2]
NUM_ANCHORS_PER_LOC= len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
BATCH_SIZE= 32
LEARNING_RATE= 1e-5
NUM_EPOCHS= 10
IOU_HIGH_THRESHOLD= 0.6
IOU_LOW_THRESHOLD= 0.5
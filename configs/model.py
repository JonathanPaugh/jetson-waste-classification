import os
from os.path import exists, isfile


JETSON_FILE = '.jetson'
IS_JETSON = exists(JETSON_FILE) and isfile(JETSON_FILE)

DATASET_TRAIN_PATH = 'dataset/train'
DATASET_TEST_PATH = 'dataset/test'
OUTPUT_PATH = 'dist'

SEED = 0
VALIDATION_SPLIT = 0.2
IMAGE_SIZE = (96, 96)

MODEL_EARLY_STOPPING_MONITOR = 'val_accuracy'
MODEL_EARLY_STOPPING_PATIENCE = 7  # stop after x consecutive epochs with no improvement
MODEL_NUM_EPOCHS = 1
MODEL_DROPOUT_RATE = 0.5
MODEL_WORKERS = os.cpu_count() | 1
MODEL_BATCH_SIZE = 2 ** (3 if IS_JETSON else 5)
LOADER_BATCH_SIZE = 2 ** (3 if IS_JETSON else 5)

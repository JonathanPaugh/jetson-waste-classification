import os
from os.path import exists, isfile

JETSON_FILE = '.jetson'
IS_JETSON = exists(JETSON_FILE) and isfile(JETSON_FILE)

DATASET_TRAIN_PATH = 'dataset/train'
DATASET_TEST_PATH = 'dataset/test'
OUTPUT_PATH = 'dist'

SEED = 0
VALIDATION_SPLIT = 0.2
IMAGE_SIZE = (96, 96)  # NOTE: should be compatible with expected feature extractor inputs

MODEL_FEATURE_EXTRACTOR = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/5'
MODEL_EARLY_STOPPING_PATIENCE = 3  # stop after x consecutive epochs with no improvement
MODEL_NUM_EPOCHS = 50
MODEL_FINE_TUNING = True
MODEL_NUM_FINE_TUNING_EPOCHS = 50
MODEL_DROPOUT_RATE = 0.3
MODEL_WORKERS = os.cpu_count() | 1
MODEL_BATCH_SIZE = 2 ** (3 if IS_JETSON else 5)
LOADER_BATCH_SIZE = 2 ** (3 if IS_JETSON else 5)

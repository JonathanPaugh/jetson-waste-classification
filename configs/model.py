import math
import os
from os.path import exists, isfile

JETSON_FILE = '.jetson'
IS_JETSON = exists(JETSON_FILE) and isfile(JETSON_FILE)

DATASET_TRAIN_PATH = 'dataset/train'
DATASET_TEST_PATH = 'dataset/test'
OUTPUT_PATH = 'dist'

SEED = 0
VALIDATION_SPLIT = 0.2
IMAGE_SIZE = (96, 72)

MODEL_NUM_EPOCHS = 5
MODEL_WORKERS = os.cpu_count() | 1
MODEL_BATCH_SIZE = int(math.pow(2, 3)) if IS_JETSON else int(math.pow(2, 6))

LOADER_BATCH_SIZE = int(math.pow(2, 3)) if IS_JETSON else int(math.pow(2, 5))


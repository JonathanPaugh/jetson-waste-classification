import math
import os

DATASET_TRAIN_PATH = 'dataset/train'
DATASET_TEST_PATH = 'dataset/test'
OUTPUT_PATH = 'dist'

SEED = 0
VALIDATION_SPLIT = 0.2
IMAGE_SIZE = (512, 384)

MODEL_NUM_EPOCHS = 2
MODEL_WORKERS = os.cpu_count() | 1
MODEL_BATCH_SIZE = int(math.pow(2, 6))

LOADER_BATCH_SIZE = int(math.pow(2, 5))

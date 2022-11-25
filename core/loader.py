from pathlib import Path
from keras.preprocessing.image_dataset import image_dataset_from_directory
import configs.model as config


def load_dataset():
    dataset_dir = Path(config.DATASET_PATH)
    loader_params = {
        'validation_split': config.VALIDATION_SPLIT,
        'seed': config.SEED,
        'image_size': config.IMAGE_SIZE,
        'batch_size': config.LOADER_BATCH_SIZE,
    }

    train_data = image_dataset_from_directory(dataset_dir, subset='training', **loader_params)
    val_data = image_dataset_from_directory(dataset_dir, subset='validation', **loader_params)
    return train_data, val_data

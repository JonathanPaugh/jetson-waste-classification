from pathlib import Path

from tensorflow.python.keras.layers.preprocessing.image_preprocessing import get_interpolation
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory, load_image
import configs.model as config

def _build_loader_params(**kwargs):
    return {
        'seed': config.SEED,
        'image_size': config.IMAGE_SIZE,
        'batch_size': config.LOADER_BATCH_SIZE,
        **kwargs,
    }


def load_train_dataset():
    dataset_dir = Path(config.DATASET_TRAIN_PATH)
    loader_params = _build_loader_params(validation_split=config.VALIDATION_SPLIT)

    train_data = image_dataset_from_directory(dataset_dir, subset='training', **loader_params)
    val_data = image_dataset_from_directory(dataset_dir, subset='validation', **loader_params)

    return train_data, val_data


def load_test_dataset():
    dataset_dir = Path(config.DATASET_TEST_PATH)
    loader_params = _build_loader_params(labels=None)

    return image_dataset_from_directory(dataset_dir, **loader_params)


def load_image_tensor(path, image_size=config.IMAGE_SIZE, color_mode='rgb', interpolation='bilinear'):
    if color_mode == 'rgb':
        num_channels = 3
    elif color_mode == 'rgba':
        num_channels = 4
    elif color_mode == 'grayscale':
        num_channels = 1
    else:
        raise ValueError(
            '`color_mode` must be one of {"rgb", "rgba", "grayscale"}. '
            'Received: %s' % (color_mode,)
        )

    interpolation = get_interpolation(interpolation)

    return load_image(
        path,
        image_size,
        num_channels,
        interpolation,
    )

import pickle
from os.path import join, exists, isfile
import configs.model as config

DEFAULT_PREFIX = 'trained_model'
WEIGHT_EXTENSION = '.sav'
HISTORY_EXTENSION = '.hist'


def _load_weights(model, prefix):
    path = join(config.OUTPUT_PATH, f'{prefix}{WEIGHT_EXTENSION}')
    return model.load_weights(path)


def _save_weights(model, prefix):
    path = join(config.OUTPUT_PATH, f'{prefix}{WEIGHT_EXTENSION}')
    model.save_weights(path)


def _read_history(prefix):
    path = join(config.OUTPUT_PATH, f'{prefix}{HISTORY_EXTENSION}')
    with open(path, mode='rb') as file:
        history = pickle.load(file)

    return history


def _write_history(history, prefix):
    path = join(config.OUTPUT_PATH, f'{prefix}{HISTORY_EXTENSION}')
    with open(path, mode='wb') as file:
        pickle.dump(history, file)


def has_trained_model(prefix=DEFAULT_PREFIX):
    weight_index_path = join(config.OUTPUT_PATH, f'{prefix}{WEIGHT_EXTENSION}.index')
    history_path = join(config.OUTPUT_PATH, f'{prefix}{HISTORY_EXTENSION}')
    return exists(weight_index_path) and isfile(weight_index_path) \
        and exists(history_path) and isfile(history_path)


def import_trained_model(model, prefix=DEFAULT_PREFIX):
    _load_weights(model, prefix)
    history = _read_history(prefix)
    print(f'Trained model imported with prefix: {prefix}')
    return history


def export_trained_model(model, history, prefix=DEFAULT_PREFIX):
    _save_weights(model, prefix)
    _write_history(history, prefix)
    print(f'Trained model exported with prefix: {prefix}')

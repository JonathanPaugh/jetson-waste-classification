from os import mkdir, listdir
from os.path import join, exists, isfile
import configs.model as config


def has_saved_weights():
    return exists(config.PICKLE_PATH) and not isfile(config.PICKLE_PATH)\
           and len(listdir(config.PICKLE_PATH))


def serialize_weights(model, checkpoint_name):
    try:
        mkdir(config.PICKLE_PATH)
    except FileExistsError:
        pass

    ckpt_path = join(config.PICKLE_PATH, checkpoint_name)
    model.save_weights(ckpt_path)
    print(f'Wrote weights to {ckpt_path}')

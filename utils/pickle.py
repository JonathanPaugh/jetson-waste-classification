from os import mkdir, listdir
from os.path import join, exists, isfile
import configs.model as config


def has_saved_weights():
    return exists(config.OUTPUT_PATH) and not isfile(config.OUTPUT_PATH)\
           and len(listdir(config.OUTPUT_PATH))


def load_weights(model, checkpoint_name='ckpt'):
    ckpt_path = join(config.OUTPUT_PATH, checkpoint_name)
    return model.load_weights(ckpt_path)


def serialize_weights(model, checkpoint_name='ckpt'):
    try:
        mkdir(config.OUTPUT_PATH)
    except FileExistsError:
        pass

    ckpt_path = join(config.OUTPUT_PATH, checkpoint_name)
    model.save_weights(ckpt_path)
    print(f'Wrote weights to {ckpt_path}')

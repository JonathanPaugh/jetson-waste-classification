from tensorflow.python.framework.config import set_memory_growth, list_physical_devices

from configs.jetson import GPU_ASYNC_MALLOC, GPU_MEMORY_GROWTH
from os import environ
from os.path import exists, isfile

JETSON_FILE = ".jetson"

def is_jetson():
    return exists(JETSON_FILE) and isfile(JETSON_FILE)

def get_gpu():
    return list_physical_devices('GPU')[0]

def tweak_hardware_settings():
    if not is_jetson():
        return

    if GPU_ASYNC_MALLOC:
        environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    if GPU_MEMORY_GROWTH:
        device_gpu = get_gpu()
        set_memory_growth(device_gpu, True)

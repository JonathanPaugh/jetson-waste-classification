from tensorflow.python.framework.config import set_memory_growth, list_physical_devices
from configs.model import IS_JETSON
from configs.jetson import GPU_ASYNC_MALLOC, GPU_MEMORY_GROWTH
from os import environ

def get_gpu():
    return list_physical_devices('GPU')[0]

def tweak_hardware_settings():
    if not IS_JETSON:
        return

    if GPU_ASYNC_MALLOC:
        environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    if GPU_MEMORY_GROWTH:
        device_gpu = get_gpu()
        set_memory_growth(device_gpu, True)

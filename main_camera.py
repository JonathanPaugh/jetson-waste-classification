import cv2
from datetime import datetime
from os import path
from configs.model import OUTPUT_PATH
from utils_jetson import sensor_camera
from utils_jetson.hardware import tweak_hardware_settings

while not input('Press enter to take snapshot'):
    tweak_hardware_settings()

    image = sensor_camera.snapshot()
    file = path.join(OUTPUT_PATH, f'snapshot-{datetime.now()}.png')
    cv2.imwrite(file, image.numpy())
    print(f'Wrote snapshot: {file}')

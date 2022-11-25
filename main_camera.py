from datetime import datetime
from os import path

import cv2

from configs.model import OUTPUT_PATH
from utils_jetson import sensor_camera

while not input("Press enter to take snapshot"):
    image = sensor_camera.snapshot()
    file = path.join(OUTPUT_PATH, f"snapshot-{datetime.now()}.png")
    cv2.imwrite(file, image)
    print(f"Wrote snapshot: {file}")

import os
from jetson_utils import videoSource, saveImage, cudaDeviceSynchronize
from os import path
from configs.model import IMAGE_SIZE, OUTPUT_PATH
import cv2

DEFAULT_WIDTH, DEFAULT_HEIGHT = IMAGE_SIZE
TEMP_FILE = path.join(OUTPUT_PATH, "temp.png")

def _get_camera():
    return videoSource("/dev/video0")


def snapshot(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, format="rgb8"):
    """

    :param width: Width to capture
    :param height: Height to capture
    :param format: {rgb8|rgba8|rgb32f|rgba32f}
    :return:
    """

    camera = videoSource("/dev/video0", argv=[
        f"--input-width={width}",
        f"--input-height={height}",
    ])

    camera.Open()

    capture = None
    if camera.IsStreaming():
        capture = camera.Capture(format=format)
        camera.Close()

    cudaDeviceSynchronize()
    saveImage(TEMP_FILE, capture)

    del capture

    pixels = cv2.imread(TEMP_FILE)
    os.remove(TEMP_FILE)

    return pixels

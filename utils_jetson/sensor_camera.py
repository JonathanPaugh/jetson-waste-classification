import os
from jetson_utils import videoSource, saveImage, cudaDeviceSynchronize
from os import path
from configs.model import IMAGE_SIZE, OUTPUT_PATH
from core.loader import load_image_tensor

VIDEO_PATH = "/dev/video0"
TEMP_FILE = path.join(OUTPUT_PATH, "temp.png")
DEFAULT_WIDTH, DEFAULT_HEIGHT = IMAGE_SIZE

def _get_camera(args):
    return videoSource(VIDEO_PATH, argv=args)

def snapshot(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, format="rgb8"):
    """
    Opens the camera stream to take a single image capture.

    The width and height will not set capture to the exact values,
    it will output the closest possible resolution.

    :param width: Width to capture
    :param height: Height to capture
    :param format: {rgb8|rgba8|rgb32f|rgba32f}
    :return: Numpy array of pixel values
    """

    camera = _get_camera([
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

    image = load_image_tensor(TEMP_FILE)
    os.remove(TEMP_FILE)

    return image

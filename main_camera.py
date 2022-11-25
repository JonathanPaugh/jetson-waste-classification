from cv2 import imshow, waitKey
from nanocamera import Camera
from configs.model import IMAGE_SIZE

WIDTH, HEIGHT = IMAGE_SIZE

camera = Camera(camera_type=1, device_id=0, width=WIDTH, height=HEIGHT)

while camera.isReady():
    try:
        # read the camera image
        frame = camera.read()
        # display the frame
        imshow("Video Frame", frame)
        print("HI")
        if waitKey(25) & 0xFF == ord('q'):
            break
    except KeyboardInterrupt:
        break

    # close the camera instance
camera.release()

# remove camera object
del camera

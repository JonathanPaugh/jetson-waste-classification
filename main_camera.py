import jetson.utils

camera = jetson.utils.videoSource("/dev/video0")

frame = camera.Capture()

print(frame)

input("Press any key...")

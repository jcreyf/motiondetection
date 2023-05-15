#!/usr/bin/env python3

# From:
#   https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html
# Resolutions:
#   https://www.learnpythonwithrune.org/find-all-possible-webcam-resolutions-with-opencv-in-python/

# pip install -r requirements.txt
import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Show a color video feed:
    cv.imshow('frame', frame)
    # To show a grayscale feed:
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #cv.imshow('frame', gray)

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cam.release()
cv.destroyAllWindows()
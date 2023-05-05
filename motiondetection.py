#!/usr/bin/env python3
# ======================================================================================================== #
# App to try detect motion in a camera feed.                                                               #
#                                                                                                          #
# Arguments:                                                                                               #
#    --version                         : show app version                                                  #
#    -v          | --verbose           : show verbose level output                                         #
#                                                                                                          #
# Example:                                                                                                 #
#   /> $0 -v                                                                                               #
#                                                                                                          #
# ======================================================================================================== #
#  2023-05-04  v0.1  jcreyf  Initial version.                                                              #
# ======================================================================================================== #

# ToDo:
#   - detect areas where movements can be seen;
#   - try to zoom into those areas and take snapshots;
#   - start recording video when movement is detected and stop a few seconds after last movement;
#   - send a message with the base image and zoomed in pictures.  Also add link to where video will be stored;

# pip install -r requirements.txt
import cv2
import numpy as np
#from PIL import ImageGrab


def motiondetection():
    previous_frame = None

    # https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture an image from the camera:
        # RPi: https://pillow.readthedocs.io/en/stable/reference/ImageGrab.html
        ret_val, img_brg = cam.read() #cam.read() returns ret (0/1 if the camera is working) and img_brg, the actual image of the camera in a numpy array
        # if frame is read correctly ret is True
        if not ret_val:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert the image to RGB:
        img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)

        # Convert the image; grayscale and blur
        prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

        # Calculate the difference
        if previous_frame is None:
            # First frame; there is no previous one yet
            previous_frame = prepared_frame
            continue

        # Calculate difference and update previous frame
        diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
        previous_frame = prepared_frame

        # Dilute the image a bit to make differences more seeable; more suitable for contour detection
        kernel = np.ones((5, 5))
        diff_frame = cv2.dilate(diff_frame, kernel, 1)

        # Only take different areas that are different enough (>20 / 255)
        thresh_frame = cv2.threshold(
            src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY
        )[1]

        # Find and optionally draw contours
        contours, _ = cv2.findContours(
            image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )
        # Comment below to stop drawing contours
        cv2.drawContours(
            image=img_rgb,
            contours=contours,
            contourIdx=-1,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        # Uncomment 6 lines below to stop drawing rectangles
        # for contour in contours:
        #   if cv2.contourArea(contour) < 50:
        #     # too small: skip!
        #       continue
        #   (x, y, w, h) = cv2.boundingRect(contour)
        #   cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

        cv2.imshow("Motion detector", img_rgb)

#        if cv2.waitKey(30) == 27:
#            # out.release()
#            break
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    motiondetection()

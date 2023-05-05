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
from datetime import datetime

class MotionDetector:
    __version__ = "v0.1 - 2023-05-04"

    @classmethod
    def version(cls) -> str:
        """ Static app version details """
        return f"{cls.__name__}: {cls.__version__}"


    def __init__(self) -> None:
        """ Constructor, initializing properties with default values. """
        self._previous_frame = None
        self._camera_number = 0
        self._camera = None
        self._opencv_diffing_threshold = 20


    @property
    def diffingThreshold(self) -> int:
        return self._opencv_diffing_threshold

    @diffingThreshold.setter
    def diffingThreshold(self, value: int) -> None:
        self._opencv_diffing_threshold = value


    def listCameras(self):
        pass


    def loadConfig(self) -> None:
        pass


    def sendAlert(self) -> None:
        pass


    def log(self, msg: str) -> None:
        """ Method to log messages.

        We have to assume that this process may be running in the background and that output is piped to
        a log-file.  Because of that, make sure we flush the stdout buffer to keep tails in sync with the
        real world.
        """
        print(f"{datetime.now().strftime('%m/%d %H:%M:%S')}: {msg}", flush=True)


    def logDebug(self, msg: str) -> None:
        if self.debug:
            self.log(f"DEBUG: {msg}")


    def doIt(self) -> None:
        # https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html
        self._camera = cv2.VideoCapture(self._camera_number)
        if not self._camera.isOpened():
            self.log("Cannot open camera")
            exit()
        while True:
            # Capture an image from the camera:
            # RPi: https://pillow.readthedocs.io/en/stable/reference/ImageGrab.html
            ret_val, img_brg = self._camera.read() #cam.read() returns ret (0/1 if the camera is working) and img_brg, the actual image of the camera in a numpy array
            # if frame is read correctly ret is True
            if not ret_val:
                self.log("Can't receive frame (stream end?). Exiting ...")
                break

            # Convert the image to RGB:
            img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)
            # Convert the image; grayscale and blur
            prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

            # Calculate the difference
            if self._previous_frame is None:
                # First frame; there is no previous one yet
                self._previous_frame = prepared_frame
                continue

            # Calculate difference and update previous frame
            diff_frame = cv2.absdiff(src1=self._previous_frame, src2=prepared_frame)
            self._previous_frame = prepared_frame
            # Dilute the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))
            diff_frame = cv2.dilate(diff_frame, kernel, 1)
            # Only take different areas that are different enough (>20 / 255)
            thresh_frame = cv2.threshold(
                src=diff_frame, thresh=self._opencv_diffing_threshold, maxval=255, type=cv2.THRESH_BINARY
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
        # Stop the streaming when the user presses the "q" key:
        self.stop()


    def stop(self):
        if not self._camera is None:
            # When everything done, release the capture
            self.log("Stopping the camera stream...")
            self._camera.release()
            cv2.destroyAllWindows()

# -----

if __name__ == "__main__":
    # Run this code when this file is opened as an application:
    motion_detector = None

    def signal_handler(signum, frame):
        """ Handle CRTL+C and other kill events """
        print("Killing the app...")
        if not motion_detector is None:
            motion_detector.stop()
        exit(0)

    # Set signal handlers to deal with CTRL+C presses and other ways to kill this process.
    # We do this to close the web browser window and cleanup resources:
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Define the command-line arguments that the app supports:
    import argparse
    parser=argparse.ArgumentParser(description="Detect motion in a camera feed.")
    parser.add_argument("--version", \
                            action="version", \
                            version=MotionDetector.__version__)
    # Parse the command-line arguments:
    __ARGS=parser.parse_args()

    # Start the app:
    print(f"Starting {MotionDetector.version()}")
    motion_detector = MotionDetector()
    print("Press the 'q' key to stop the app")
    motion_detector.doIt()
    print("Ending the app")

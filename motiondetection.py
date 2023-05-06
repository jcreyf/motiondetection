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
#   - try to zoom into those areas and take snapshots;
#   - start recording video when movement is detected and stop a few seconds after last movement;
#   - send a message with the base image and zoomed in pictures.  Also add link to where video will be stored;
#   - pub/sub actions (have an abstract class that implements the basics (logging to console))
# https://stackoverflow.com/questions/57195852/eliminate-or-ignore-all-small-or-overlapping-contours-or-rectangles-inside-a-big

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
        self._debug = False
        self._previous_frame = None
        self._camera_number = 0
        self._camera = None
        self._opencv_diffing_threshold = 20
        self._minimum_pixel_difference = 50


    def __del__(self) -> None:
        """ Destructor to close the camera stream if we have one open. """
        if not self._camera is None:
            self._camera.release()
            cv2.destroyAllWindows()


    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, flag: bool) -> None:
        self._debug = flag
        if flag:
            self.logDebug("Debugging enabled")


    @property
    def diffingThreshold(self) -> int:
        return self._opencv_diffing_threshold

    @diffingThreshold.setter
    def diffingThreshold(self, value: int) -> None:
        self._opencv_diffing_threshold = value


    @property
    def minimumPixelDifference(self) -> int:
        return self._minimum_pixel_difference

    @minimumPixelDifference.setter
    def minimumPixelDifference(self, value: int) -> None:
        if value < 0 or value > 255:
            raise ValueError(f"The minimum pixel difference needs to be between 0 and 255!  You're setting: {value}")
        self._minimum_pixel_difference = value


    def logSettings(self) -> None:
        self.log("Settings:")
        self.log(f"  Debug: {self._debug}")
        self.log(f"  Minimum block pixel difference: {self._minimum_pixel_difference}")
        self.log(f"  Diffing threshold: {self._opencv_diffing_threshold}")


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
        self.logSettings()
        self.log("Starting the camera stream...")
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

            # There's nothing to detect if this is the first frame:
            if self._previous_frame is None:
                self._previous_frame = prepared_frame
                continue

            # Calculate differences between this and previous frame:
            diff_frame = cv2.absdiff(src1=self._previous_frame, src2=prepared_frame)
            self._previous_frame = prepared_frame
            # Dilute the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))
            diff_frame = cv2.dilate(diff_frame, kernel, 1)
            # Only take different areas that are different enough:
            thresh_frame = cv2.threshold(
                src=diff_frame, thresh=self._opencv_diffing_threshold, maxval=255, type=cv2.THRESH_BINARY
            )[1]

            # Reset the list of unique areas (this is used to filter out overlapping areas):
            areaList = []
            # Find contours of places in the image that has changes:
            contours, _ = cv2.findContours(
                image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
            )
            # No need to process the image if OpenCV didn't find differences:
            if len(contours) > 0:
#                self.logDebug(f"found {len(contours)} differences")
                # Draw contours around the changed areas:
#                cv2.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                # Draw rectangles around the changed areas:
                for contour in contours:
                    if cv2.contourArea(contour) < self._minimum_pixel_difference:
                        # The dectected difference area is too small.  Lets ignore it.
#                        self.logDebug(f"change area is too small for our filter: {cv2.contourArea(contour)}")
                        continue
                    # Get the coordinates of the contour and add them to an array that we will use to find the global change window:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    areaList.append(list((x, y, x+w, y+h)))
                    if self.debug:
                        # Draw a thin rectangle arround all found differences.
                        # We're not really interested in seeing all those individual differences though.
                        cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=1)

            # Find the smallest top left corner and the largest bottom right corner of all contours.
            # All changes fall within those coordinates:
            if len(areaList) > 0:
                # This works:
#                _x1=[min(i) for i in zip(*areaList)][0]
#                _y1=[min(i) for i in zip(*areaList)][1]
#                _x2=[max(i) for i in zip(*areaList)][2]
#                _y2=[max(i) for i in zip(*areaList)][3]
                # Using list comprehension is cleaner and faster?:
                _x1=min(map(lambda x: x[0], areaList))
                _y1=min(map(lambda x: x[1], areaList))
                _x2=max(map(lambda x: x[2], areaList))
                _y2=max(map(lambda x: x[3], areaList))
                # Now draw a thick rectangle around the full change window:
                cv2.rectangle(img=img_rgb, pt1=(_x1, _y1), pt2=(_x2, _y2), color=(0, 255, 0), thickness=2)

            # Show the processed picture in a window:
            cv2.imshow("Motion detector", img_rgb)
            # Keep iterating through this while loop until the user presses the "q" button.
            # The app that is wrapping around this class can also have a signal handler to deal with <CTRL><C> or "kill" commands.
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
    parser.add_argument("--debug", "-d", \
                            action="store_true", \
                            help="Turn on debug-level logging")
    parser.add_argument("-pd", "--pixeldiff", \
                            default=100, \
                            dest="__PD", \
                            required=False, \
                            type=int, \
                            help="Minimum number of pixels in the same block that need to be different [default=100]")
    parser.add_argument("-dt", "--diffingthreshold", \
                            default=20, \
                            dest="__DT", \
                            required=False, \
                            type=int, \
                            help="Diffing threshold (sensitivity) [default=20]")
    # Parse the command-line arguments:
    __ARGS=parser.parse_args()

    # Start the app:
    print(f"Starting {MotionDetector.version()}")
    motion_detector = MotionDetector()
    motion_detector.debug = __ARGS.debug
    motion_detector.diffingThreshold = __ARGS.__DT
    motion_detector.minimumPixelDifference = __ARGS.__PD
    print("Press the 'q' key to stop the app (the camera window needs to have the focus!)")
    motion_detector.doIt()
    print("Ending the app")

#!/usr/bin/env python3
# ======================================================================================================== #
# App to try detect motion in a camera feed.                                                               #
#                                                                                                          #
# Arguments:                                                                                               #
#    --version                         : show app version                                                  #
#    -v          | --verbose           : show verbose level output                                         #
#                                                                                                          #
# Example:                                                                                                 #
#   /> $0 ./motiondetection.py -sv -clr '{"w": 320, "h": 240}' -chr '{"w": 1920, "h": 1080}'               #
#                                                                                                          #
# ======================================================================================================== #
#  2023-05-04  v0.1  jcreyf  Initial version.                                                              #
# ======================================================================================================== #

# ToDo:
#   - try to zoom into those areas and take snapshots;
#   - start recording video when movement is detected and stop a few seconds after last movement;
#   - save raw file and add contour rectangle to image fields;
#   - send a message with the base image and zoomed in pictures.  Also add link to where video will be stored;
#   - pub/sub actions (have an abstract class that implements the basics (logging to console))
# https://stackoverflow.com/questions/57195852/eliminate-or-ignore-all-small-or-overlapping-contours-or-rectangles-inside-a-big
# https://stackoverflow.com/questions/29664399/capturing-video-from-two-cameras-in-opencv-at-once
# https://www.raspberrypi.com/documentation/accessories/camera.html#:~:text=About%20the%20Camera%20Modules,-Edit%20this%20on&text=There%20are%20now%20several%20official,which%20was%20released%20in%202023.
# https://projects.raspberrypi.org/en/projects/getting-started-with-picamera/7


# pip install -r requirements.txt
import os
import json
import cv2
import numpy as np
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
        self._showVideo = False
        self._previous_frame = None
        self._camera_port_number = 0
        self._camera = None
        self._camera_resolution_default = {"w":0, "h":0}
        self._camera_resolution_low = {"w":640, "h":480}
        self._camera_resolution_high = {"w":1920, "h":1080}
        self._camera_rotation = 0
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
    def showVideo(self) -> bool:
        return self._showVideo

    @showVideo.setter
    def showVideo(self, flag: bool) -> None:
        self._showVideo = flag
        if flag:
            self.logDebug("Showing video stream on screen")


    @property
    def cameraPortNumber(self) -> int:
        return self._camera_port_number

    @cameraPortNumber.setter
    def cameraPortNumber(self, value: int) -> None:
        self._camera_port_number = value


    @property
    def cameraLowResolution(self):
        return self._camera_resolution_low

    @cameraLowResolution.setter
    def cameraLowResolution(self, value) -> None:
        self._camera_resolution_low=value


    @property
    def cameraHighResolution(self):
        return self._camera_resolution_high

    @cameraHighResolution.setter
    def cameraHighResolution(self, value) -> None:
        self._camera_resolution_high=value


    @property
    def cameraRotation(self) -> int:
        return self._camera_rotation

    @cameraPortNumber.setter
    def cameraRotation(self, value: int) -> None:
        self._camera_rotation = value


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
        self.log(f"  Show video stream: {self._showVideo}")
        self.log(f"  Minimum block pixel difference: {self._minimum_pixel_difference}")
        self.log(f"  Diffing threshold: {self._opencv_diffing_threshold}")
        self.log(f"  Camera resolution (default): {self._camera_resolution_default}")
        self.log(f"  Camera resolution (low): {self._camera_resolution_low}")
        self.log(f"  Camera resolution (high): {self._camera_resolution_high}")
        self.log(f"  Camera rotation: {self._camera_rotation} degrees")


    def listCameras(self):
        """
        Test the ports and returns a tuple with the available ports and the ones that are working.
        export OPENCV_LOG_LEVEL=OFF
        export OPENCV_LOG_LEVEL=DEBUG
        export OPENCV_VIDEOIO_DEBUG=1
        """
        non_working_ports = []
        device_port = 0
        working_ports = []
        available_ports = []
        # if there are more than 5 non working ports stop the testing:
#        os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
        while len(non_working_ports) < 6:
            camera = cv2.VideoCapture(device_port)
            if not camera.isOpened():
                non_working_ports.append(device_port)
                self.logDebug(f"Port {device_port} is not working.")
            else:
                is_reading, img = camera.read()
                w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                camera.release()
                if is_reading:
                    self.logDebug(f"Port {device_port} is working and reads images ({w} x {h})")
                    working_ports.append({"port":device_port, "w":w, "h":h})
                else:
                    self.logDebug(f"Port {device_port} for camera ({w} x {h}) is present but does not reads.")
                    available_ports.append(device_port)
            device_port +=1
        return available_ports, working_ports, non_working_ports


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


    def darknessScale(self, image) -> int:
        """ Return the darness level of an image.
            This is used to detect day vs. night and cloudy moments.
            We can use the value of this to make the motion detection sensitivity more dynamic.
        """
# https://stackoverflow.com/questions/52505906/find-if-image-is-bright-or-dark
        # Convert the mean to a percentage ( 0% == black; 100% == white)
        return int(np.mean(image) / 255 * 100)
#        return int(np.mean(image))

    def averageBrightness(self, image) -> int:
# https://github.com/arunnthevapalan/day-night-classifier/blob/master/classifier.ipynb
        hsv=cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        sum_brightness = np.sum(hsv[:,:,2])
        # How many pixels in an image from this camera?
        area = self._camera_resolution_low["w"] * self._camera_resolution_low["h"]
        avg=sum_brightness/area
        return int(avg)


    def lowres(self) -> None:
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(self._camera_resolution_low["w"]))
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self._camera_resolution_low["h"]))

    def highres(self) -> None:
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(self._camera_resolution_high["w"]))
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self._camera_resolution_high["h"]))


    def doIt(self) -> None:
        self.log("Starting the camera stream...")
#        scale = 100
        # https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html
        self._camera = cv2.VideoCapture(self._camera_port_number)
        if not self._camera.isOpened():
            self.log("Cannot open camera")
            exit()

        # Grab images at the highest resolution the camera supports:
        self._camera_resolution_default={"w": int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        self.highres()
        self.logSettings()
        self.log(f"Camera resolution set: {int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

        imageCnt=0

        while True:
            # Capture an image from the camera (the image is captured as a numpi array):
            # RPi: https://pillow.readthedocs.io/en/stable/reference/ImageGrab.html
            ret_val, img_brg = self._camera.read()
            # if frame is read correctly ret is True
            if not ret_val:
                self.log("Can't receive frame (stream end?)")
                # Exit the loop if this is the very time we're trying to grab an image.
                # Keep going otherwise!
                if self._previous_frame is None:
                    break

            # Rotate the image if needed:
            if self._camera_rotation > 0:
                if self._camera_rotation == 90:
                    img_brg=cv2.rotate(img_brg, cv2.ROTATE_90_CLOCKWISE)
                if self._camera_rotation == 180:
                    img_brg=cv2.rotate(img_brg, cv2.ROTATE_180)
                if self._camera_rotation == 270:
                    img_brg=cv2.rotate(img_brg, cv2.ROTATE_90_COUNTERCLOCKWISE)

            try:
                # Scale down and convert the image to RGB.
                # This sometimes fails with this error:
                #   OpenCV(4.5.3) .../resize.cpp:4051: error: (-215:Assertion failed) !ssize.empty() in function 'resize'
                # It happens because of issues with the image.
                # Lets catch the issue and ignore this image if it happens and go to the next...
                img_rgb = cv2.resize(src=img_brg, dsize=(self._camera_resolution_low["w"], self._camera_resolution_low["h"]), interpolation = cv2.INTER_AREA)
            except Exception as ex:
                # Yep, this image is bad.  Skip and go on to the next!
                self.log(f"Image resize failed! {str(e)}")
                continue

            # The resize was fine.  Keep processing it:
            img_rgb = cv2.cvtColor(src=img_rgb, code=cv2.COLOR_BGR2RGB)
            # Convert the image; grayscale and blur
            prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

            # There's nothing to detect if this is the first frame:
            if self._previous_frame is None:
                self._previous_frame = prepared_frame
                continue

# We should run this once a minute or so instead of based on image count and use this value to adjust the motion sensitivity (diffing threshold)
# Detection might be more robust if we calculate for each image and take a mean value once per minute or so to set the detection sensitivity.
            imageCnt=imageCnt+1
            if imageCnt > 50:
                imageCnt=0
                self.logDebug(f"image darkness: {self.darknessScale(prepared_frame)}")
                self.logDebug(f"image brightness: {self.averageBrightness(img_brg)}")

            # Calculate differences between this and previous frame:
            diff_frame = cv2.absdiff(src1=self._previous_frame, src2=prepared_frame)
            self._previous_frame = prepared_frame
            # Dilute the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))
            diff_frame = cv2.dilate(diff_frame, kernel, 1)
            # Only take different areas that are different enough:
            thresh_frame = cv2.threshold(src=diff_frame, thresh=self._opencv_diffing_threshold, maxval=255, type=cv2.THRESH_BINARY)[1]
            # There are ways to remove "noise" (small blobs)
            # This does not seem to work too well, so we're now ignoring all contours that are smaller than some configurable size.
# https://pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
#            thresh_frame = cv2.erode(thresh_frame, None, iterations=2)
#            thresh_frame = cv2.dilate(thresh_frame, None, iterations=4)
            if self._showVideo and self.debug:
                # Show the threshold frames if debug is enabled:
                try:
                    cv2.imshow("threshold frames", thresh_frame)
                except Exception:
                    self.log("We're getting an error when trying to display the video stream!!!")
                    self.log("Disabling trying to show the video stream!")
                    self.showVideo=False

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
                # Save this image to a file:
                filename=f"/tmp/motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                cv2.imwrite(f"{filename}.jpg", img_rgb)
                cv2.imwrite(f"{filename}_full.jpg", img_brg)

# https://stackoverflow.com/questions/50870405/how-can-i-zoom-my-webcam-in-open-cv-python
#            height, width, channels = img_brg.shape
#            centerX, centerY = int(height/2), int(width/2)
#            radiusX, radiusY = int(scale*height/100), int(scale*width/100)
#            minX, maxX = centerX-radiusX, centerX+radiusX
#            minY, maxY = centerY-radiusY, centerY+radiusY
#
#            cropped = img_brg[minX:maxX, minY:maxY]
#            resized_cropped = cv2.resize(cropped, (width, height))

            # Show the processed picture in a window if we have the flag enbabled to show the video stream:
            if self._showVideo:
                try:
#                    cv2.imshow("Motion detector", resized_cropped)
                    cv2.imshow("Motion detector", img_rgb)
#                    if cv2.waitKey(1) == ord('a'):
#                        scale += 5
#                    if cv2.waitKey(1) == ord('s'):
#                        scale -= 5
                    # Keep iterating through this while loop until the user presses the "q" button.
                    # The app that is wrapping around this class can also have a signal handler to deal with <CTRL><C> or "kill" commands.
                    if cv2.waitKey(1) == ord('q'):
                        break
                except Exception:
                    self.log("We're getting an error when trying to display the video stream!!!")
                    self.log("Disabling trying to show the video stream!")
                    self.showVideo=False

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
    os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
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
    parser.add_argument("-sv", "--showvideo", \
                            action="store_true", \
                            help="Show the video stream")
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
    parser.add_argument("-cp", "--cameraport", \
                            default=-1, \
                            dest="__CAMPORT", \
                            required=False, \
                            type=int, \
                            help="The camera to use.  The app will try to auto-detect if not set.")
    parser.add_argument("-cr", "--camerarotation", \
                            default=0, \
                            dest="__CAMROTATION", \
                            required=False, \
                            type=int, \
                            help="The camera image rotation (0 (default), 90, 180 or 270).")
    parser.add_argument("-chr", "--camerahighres", \
                            default='{"w": 1920, "h": 1080}', \
                            dest="__CAMHIGHRES", \
                            required=False, \
                            type=json.loads, \
                            help="The camera high resolution (default: 1920x1080)")
    parser.add_argument("-clr", "--cameralowres", \
                            default='{"w": 640, "h": 480}', \
                            dest="__CAMLOWRES", \
                            required=False, \
                            type=json.loads, \
                            help="The camera low resolution (default: 640x480)")
    # Parse the command-line arguments:
    __ARGS=parser.parse_args()

    # Start the app:
    print(f"Starting {MotionDetector.version()}")
    motion_detector = MotionDetector()
    # Get a list of detected cameras:
    available_ports, working_ports, nonworking_ports = motion_detector.listCameras()
    if __ARGS.debug:
        print(f"Cameras:\n {working_ports}")

    if len(working_ports) > 0:
        if __ARGS.__CAMPORT >= 0:
            camera = __ARGS.__CAMPORT
        else:
            camera=working_ports[0]["port"]
            print(f"Using camera {working_ports[0]['port']} with resolution: {working_ports[0]['w']}x{working_ports[0]['h']}")
        motion_detector.cameraPortNumber = camera
        motion_detector.cameraRotation = __ARGS.__CAMROTATION
        motion_detector.cameraLowResolution = __ARGS.__CAMLOWRES
        motion_detector.cameraHighResolution = __ARGS.__CAMHIGHRES
        motion_detector.debug = __ARGS.debug
        motion_detector.diffingThreshold = __ARGS.__DT
        motion_detector.minimumPixelDifference = __ARGS.__PD
        motion_detector.showVideo = __ARGS.showvideo
        print("Press the 'q' key to stop the app (the camera window needs to have the focus!)")
        motion_detector.doIt()
        print("Ending the app")
    else:
        print("It looks like there's no camera available!")

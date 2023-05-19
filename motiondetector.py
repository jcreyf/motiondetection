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
#  2023-05-15  v0.2  jcreyf  Adding configuration file support with hot reloads                            #
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
import os                       # Used to get hostname
import re                       # Used to validate dynamic exclusion zone values (regex to check and parse values)
import sys                      # Used to return exit values
import time                     # Used to work with the timestamp the config-file was changed
import yaml                     # Used to load and parse the YAML config-file
import ast                      # Used to remove comment lines from the schema definition file
import pprint                   # Used to pretty-print the config;
import cv2                      # The heart of the app ... OpenCV
import numpy as np              # Used to manipulate OpenCV images (they are stored as numpy arrays)
from datetime import datetime   # Used to generate timestamps
from cerberus import Validator  # Needed to validate and normalize the config-file

class MotionDetector:
    __version__ = "v0.2 - 2023-05-15"

    @classmethod
    def version(cls) -> str:
        """ Static app version details """
        return f"{cls.__name__}: {cls.__version__}"


    def __init__(self) -> None:
        """ Constructor, initializing properties with default values. """
        self._needToReload = False          # A flag that is set when we detect that the config-file has changed;
        self._configMode = False            # Is the app running in config mode? (used to validate the config-file and to set/test the exclusion zones);
        self._configFile = None             # Full path to the config-file;
        self._configDate = None             # Modification date of the config-file;
        self._camera = None                 # The camera object that we use to capture images from;
        self._camera_resolution_x = 0       # The width of the current camera resultion
        self._camera_resolution_y = 0       # The height of the current camera resoltion
        self._image = None                  # The current image
        self._previous_image = None         # A copy of the previous image (this is what we use to compare the new image against);
        self._settings = {                  # Dictionary with our settings loaded from the config-file;
            "debug": False,
            "show_video": False,
            "diffing_threshold": 20,
            "min_pixel_diff": 100,
            "image_directory": "/tmp",
            "cameras": [{
                "name": "cam0",
                "port_number": 0,
                "rotation": 0,              # 0, 90, 180 or 270 degrees;
                "low_res": {"width": 640, "height": 480},
                "high_res": {"width": 1280, "height": 720}
            }],
            "exclusion_zones": [],
            "notifications": []
        }


    def __del__(self) -> None:
        """ Destructor to close the camera stream if we have one open. """
        if not self._camera is None:
            self._camera.release()
            cv2.destroyAllWindows()


    @property
    def needToReload(self) -> bool:
        return self._needToReload


    @property
    def configMode(self) -> bool:
        return self._configMode

    configMode.setter
    def configMode(self, flag: bool) -> None:
        self._configMode = flag


    @property
    def configFile(self) -> str:
        return self._configFile

    @configFile.setter
    def configFile(self, path: str) -> None:
        self._configFile = path


    @property
    def configDate(self) -> time:
        return self._configDate


    @property
    def debug(self) -> bool:
        return self._settings['config']['debug']

    @debug.setter
    def debug(self, flag: bool) -> None:
        self._settings['config']['debug'] = flag
        if flag:
            self.logDebug("Debugging enabled")


    @property
    def hostname(self) -> str:
        return self._settings['config']['hostname']


    @property
    def diffingThreshold(self) -> int:
        return self._settings['config']['diffing_threshold']

    @diffingThreshold.setter
    def diffingThreshold(self, value: int) -> None:
        self._settings['config']['diffing_threshold'] = value


    @property
    def minimumPixelDifference(self) -> int:
        return self._settings['config']['min_pixel_diff']

    @minimumPixelDifference.setter
    def minimumPixelDifference(self, value: int) -> None:
        if value < 0 or value > 255:
            raise ValueError(f"The minimum pixel difference needs to be between 0 and 255!  You're setting: {value}")
        self._settings['config']['min_pixel_diff'] = value


    @property
    def showVideo(self) -> bool:
        return self._settings['config']['show_video']

    @showVideo.setter
    def showVideo(self, flag: bool) -> None:
        self._settings['config']['show_video'] = flag
        if flag:
            self.logDebug("Showing video stream on screen")


    @property
    def cameraPortNumber(self) -> int:
        return self._settings['config']['cameras'][0]['port_number']

    @cameraPortNumber.setter
    def cameraPortNumber(self, value: int) -> None:
        self._settings['config']['cameras'][0]['port_number'] = value


    @property
    def cameraName(self) -> int:
        return self._settings['config']['cameras'][0]['name']


    @property
    def cameraLowResolution(self):
        return self._settings['config']['cameras'][0]['low_res']

    @cameraLowResolution.setter
    def cameraLowResolution(self, value) -> None:
        self._settings['config']['cameras'][0]['low_res'] = value


    @property
    def cameraHighResolution(self):
        return self._settings['config']['cameras'][0]['high_res']

    @cameraHighResolution.setter
    def cameraHighResolution(self, value) -> None:
        self._settings['config']['cameras'][0]['high_res'] = value


    @property
    def cameraRotation(self) -> int:
        return self._settings['config']['cameras'][0]['rotation']

    @cameraRotation.setter
    def cameraRotation(self, value: int) -> None:
        self._settings['config']['cameras'][0]['rotation'] = value


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


    def saveProcessID(self):
        """ Method to save the PID of this process to a file.
        
        It's sometimes handy to have the process ID for command-line operation.
        We could of course get the PID through the ps-command but why no save it to be sure!?
        """
        try:
            pid_file_path = f"{os.path.dirname(os.path.realpath(__file__))}/motiondetector.pid"
            pid = os.getpid()
            with open(pid_file_path, 'w') as pid_file:
                self.log(f"PID: {pid} -> {pid_file_path}")
                pid_file.write(f"{pid}\n")
        except:
            pass


    def loadConfig(self):
        """ Method to load the config-file for this app.

        We expect the file to be called 'motiondetector.yaml' and sit in the same directory as the app.
        The file layout:
            ---
            config:
                debug: true
                show_video: true
                diffing_threshold: 20
                min_pixel_diff: 120
                image_directory: {app}/images
                cameras:
                  - name: internal
                    port: 0
                    rotation: 0
                    low_res:
                        width: 640
                        height: 480
                    high_res:
                        width: 1920
                        height: 1080
                exclusion_zones:
                  - name: top
                    top_x: 0
                    top_y: 0
                    bottom_x: {20%}
                    bottom_y: {100%}
                  - name: left_side
                    top_x: 0
                    top_y: {20%}
                    bottom_x: {100%}
                    bottom_y: 300
                notifications:
                  - enabled: false
                    email_to: "to@gmail.com"
                    email_from: "from@gmail.com"
                    app_token: "jvl..rfx"
                    password: "dC9V..iQQ=="
        """
        # Figure out this app's directory and add the name of the config-file to load:
        self.configFile = f"{os.path.dirname(os.path.realpath(__file__))}/motiondetector.yaml"
        self.log(f"Load config: {self.configFile}")
        # Load the config file:
        with open(self.configFile, "r") as stream:
            try:
                _settings = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print("Failed to read the config file!")
                print(e)
                sys.exit(1)

        # Get the modification time of the config-file.
        # We'll use this to detect file changes and dynamically reload the config when it changes.
        self._configDate = os.path.getmtime(self.configFile)
        # Load the schema definition file so that we can validate the config-file:
        # I could have used Pydantic for this but I wanted to do this with Cerberus.
        _configSchemaFile = f"{os.path.dirname(os.path.realpath(__file__))}/config.schema"
        self.log(f"Load schema definition: {_configSchemaFile}")
        with open(_configSchemaFile, 'r') as stream:
            try:
                _config_schema_definition = stream.read()
                # Remove comment-lines and turn the string into a dict:
                # (using eval() could be used too but is not secure since it can execute commands in strings!)
                _config_schema_definition = ast.literal_eval(_config_schema_definition)
            except Exception as e:
                print("Failed to read the config schema definition file!")
                print(e)
                sys.exit(1)

        # Validate the config:
        self.log(f"Validating config...")
        validator = Validator(_config_schema_definition, purge_unknown = True)
        if validator.validate(_settings):
            # The config is fine.  Normalize it to add potential missing optional settings:
            self._settings = validator.normalized(_settings)
        else:
            # The config has issues!
            print("The config has issues!!!")
            print(validator.errors)
            sys.exit(1)

        # Set the hostname if it isn't set in the config-file:
        if self.hostname == '':
            self._settings['config']['hostname'] = os.uname()[1]
        # Add an empty exclusion zones dictionary if there is none in the config-file:
        if not "exclusion_zones" in self._settings["config"]:
            self._settings["config"]["exclusion_zones"] = []
        # Add an empty notifications dictionary if there is none in the config-file:
        if not "notifications" in self._settings["config"]:
            self._settings["config"]["notifications"] = []
        # Reset the reload-flag:
        self._needToReload = False
        


    def configFileChanged(self) -> bool:
        """ Method to check if the config-file changed after starting the app.
        
        We want to auto-restart the app whenever the config-file got updated.
        """
        # Get the current modification time of the config-file and compare to what it was when we loaded the config:
        _configDate = os.path.getmtime(self.configFile)
        if _configDate != self.configDate:
            self.log("The config-file changed.  We need to reload the app!")
            self._needToReload = True
            return True


    def logSettings(self) -> None:
        self.log(f"Config:\n{pprint.pformat(self._settings)}")


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
                    working_ports.append({"port_number":device_port, "width":w, "height":h})
                else:
                    self.logDebug(f"Port {device_port} for camera ({w} x {h}) is present but does not reads.")
                    available_ports.append(device_port)
            device_port +=1
        return available_ports, working_ports, non_working_ports


    def sendAlert(self) -> None:
        pass


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
        area = self.cameraLowResolution["width"] * self.cameraLowResolution["height"]
        avg=sum_brightness/area
        return int(avg)


    def lowres(self) -> None:
        """ Switch the camera to low resolution mode """
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cameraLowResolution["width"]))
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cameraLowResolution["height"]))

    def highres(self) -> None:
        """ Switch the camera to high resolution mode """
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cameraHighResolution["width"]))
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cameraHighResolution["height"]))


    def __startCamera(self) -> None:
        """ Private method to start the camera stream. """
        self.log(f"Starting camera '{self.cameraName}' stream...")
        # https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html
        self._camera = cv2.VideoCapture(self.cameraPortNumber)
        if not self._camera.isOpened():
            self.log("Cannot open camera")
            exit()

        # Grab images at the highest resolution the camera supports:
        self.highres()
        self.logSettings()
        self._camera_resolution_x = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._camera_resolution_y = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.log(f"Camera resolution set: {self._camera_resolution_x}x{self._camera_resolution_y}")


    def __rotateImage(self) -> None:
        """ Private method to rotate the raw image. """
        if self.cameraRotation > 0:
            if self.cameraRotation == 90:
                self._image=cv2.rotate(self._image, cv2.ROTATE_90_CLOCKWISE)
            elif self.cameraRotation == 180:
                self._image=cv2.rotate(self._image, cv2.ROTATE_180)
            elif self.cameraRotation == 270:
                self._image=cv2.rotate(self._image, cv2.ROTATE_90_COUNTERCLOCKWISE)


    def __parseExclusionZoneValue(self, value: str, max: int) -> int:
        """ The exclusion zone value can be a number or a string in this format: '{NNN%}' """
        if re.search("^{\d{1,3}%}$", value):
            self.log(f"    value '{value}' is a percentage of {max} pixels ({value[1:-2]}%)")
            percentage = int(value[1:-2])
            return int(max * percentage / 100)
        else:
            return int(value)


    def __setupExclusionZones(self) -> None:
        """ Private method to read the exclusion zone config and turn it into a usable dictionary. """
        zones=self._settings["config"]["exclusion_zones"]
        if len(zones) == 0:
            self.log("No exclusion zones configured")
            return

        self.log(f"Processing {len(zones)} exclusion zones:")
        for zone in zones:
            self.log(f"- {zone}")
            name = zone["name"]
            topX = zone["top_x"]
            topY = zone["top_y"]
            bottomX = zone["bottom_x"]
            bottomY = zone["bottom_y"]
            self.log(f"  {name} = {topX}:{topY} -> {bottomX}:{bottomY}")
            topX = self.__parseExclusionZoneValue(topX, self._camera_resolution_x)
            topY = self.__parseExclusionZoneValue(topY, self._camera_resolution_y)
            bottomX = self.__parseExclusionZoneValue(bottomX, self._camera_resolution_x)
            bottomY = self.__parseExclusionZoneValue(bottomY, self._camera_resolution_y)
            self.log(f"    -> {topX}:{topY} -> {bottomX}:{bottomY}")


    def run(self) -> None:
        """ Main motion detection method """
        self.__startCamera()
#        scale = 100
        imageCnt=0

        while True:
            # Capture an image from the camera (the image is captured as a numpi array):
            # RPi: https://pillow.readthedocs.io/en/stable/reference/ImageGrab.html
            ret_val, self._image = self._camera.read()
            if not ret_val:
                self.log("Can't receive frame (stream end?)")
                # Exit the loop if this is the very time we're trying to grab an image.
                # Keep going otherwise!
                if self._previous_image is None:
                    break

            # Rotate the image if needed:
            self.__rotateImage()

            try:
                # Scale down and convert the image to RGB.
                # This sometimes fails with this error:
                #   OpenCV(4.5.3) .../resize.cpp:4051: error: (-215:Assertion failed) !ssize.empty() in function 'resize'
                # It happens because of issues with the image.
                # Lets catch the issue and ignore this image if it happens and go to the next...
                img_rgb = cv2.resize(src=self._image, dsize=(self.cameraLowResolution["width"], self.cameraLowResolution["height"]), interpolation = cv2.INTER_AREA)
            except Exception as ex:
                # Yep, this image is bad.  Skip and go on to the next!
                self.log(f"Image resize failed! {str(ex)}")
                continue

            # The resize was fine.  Keep processing it:
            img_rgb = cv2.cvtColor(src=img_rgb, code=cv2.COLOR_BGR2RGB)
            # Convert the image; grayscale and blur
            prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

            # There's nothing to detect if this is the first frame:
            if self._previous_image is None:
                self._previous_image = prepared_frame
                continue

            # Calculate differences between this and previous frame:
            diff_frame = cv2.absdiff(src1=self._previous_image, src2=prepared_frame)
            self._previous_image = prepared_frame
            # Dilute the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))
            diff_frame = cv2.dilate(diff_frame, kernel, 1)
            # Only take different areas that are different enough:
# https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
            thresh_frame = cv2.threshold(src=diff_frame, thresh=self.diffingThreshold, maxval=255, type=cv2.THRESH_BINARY)[1]
            # There are ways to remove "noise" (small blobs)
            # This does not seem to work too well, so we're now ignoring all contours that are smaller than some configurable size.
# https://pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
#            thresh_frame = cv2.erode(thresh_frame, None, iterations=2)
#            thresh_frame = cv2.dilate(thresh_frame, None, iterations=4)
            if self.showVideo and self.debug:
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
                    if cv2.contourArea(contour) < self.minimumPixelDifference:
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
                cv2.imwrite(f"{filename}_full.jpg", self._image)

# https://stackoverflow.com/questions/50870405/how-can-i-zoom-my-webcam-in-open-cv-python
#            height, width, channels = self._image.shape
#            centerX, centerY = int(height/2), int(width/2)
#            radiusX, radiusY = int(scale*height/100), int(scale*width/100)
#            minX, maxX = centerX-radiusX, centerX+radiusX
#            minY, maxY = centerY-radiusY, centerY+radiusY
#
#            cropped = self._image[minX:maxX, minY:maxY]
#            resized_cropped = cv2.resize(cropped, (width, height))

            # Show the processed picture in a window if we have the flag enbabled to show the video stream:
            if self.showVideo:
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

            # Check the brightness of the image every so often to adjust the sensitivity throughout the day.
            # At the same time, also check if the user changed settings in the config=file.  Reload the app if he did!
            imageCnt=imageCnt+1
            if imageCnt > 500:
# We should run this once a minute or so instead of based on image count and use this value to adjust the motion sensitivity (diffing threshold)
# Detection might be more robust if we calculate for each image and take a mean value once per minute or so to set the detection sensitivity.
                imageCnt=0
                self.logDebug(f"image darkness: {self.darknessScale(prepared_frame)}")
                self.logDebug(f"image brightness: {self.averageBrightness(self._image)}")
                # Check and see if the config-file got updated before we continue.
                # We need to restart with the new config if it changed!
                if self.configFileChanged():
                    # Breaking out of this loop will kick us back to the main application loop, which will restart the app:
                    break

        # Stop the streaming when the user presses the "q" key:
        self.stop()


    def runConfig(self) -> None:
        """ Run the application in config-mode. 
        
        This basically pulls part of the configuration to test things like exlusion zones and alerting.
        """
        self.log("** RUNNING IN CONFIG-MODE **")
        self.__startCamera()
        # Capture an image from the camera (the image is captured as a numpi array):
        ret_val, self._image = self._camera.read()
        if not ret_val:
            self.log("Can't receive frame (stream end?)")

        # Rotate the image if needed:
        self.__rotateImage()
        # Setup exclusion zones:
        self.__setupExclusionZones()

        # Save this image to a file:
        cv2.imwrite("/tmp/motion_config.jpg", self._image)

        self.stop()


    def stop(self):
        """ Method to stop the camera strean and clean up the application """
        if not self._camera is None:
            # When everything done, release the capture
            self.log("Stopping the camera stream...")
            self._camera.release()
            cv2.destroyAllWindows()
        # Delete the PID-file (if there):
        try:
            pid_file_path = f"{os.path.dirname(os.path.realpath(__file__))}/motiondetector.pid"
            os.remove(pid_file_path)
        except:
            # Ignore all potential issues with this
            pass


# -----

if __name__ == "__main__":
    import json         # Needed to parse optional JSON args from command line
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
    parser.add_argument("-d", "--debug", \
                            action="store_true", \
                            help="Turn on debug-level logging")
    parser.add_argument("-c", "--config", \
                            action="store_true", \
                            help="Running in config-mode")
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
                            default='{"width": 1920, "height": 1080}', \
                            dest="__CAMHIGHRES", \
                            required=False, \
                            type=json.loads, \
                            help="The camera high resolution (default: 1920x1080)")
    parser.add_argument("-clr", "--cameralowres", \
                            default='{"width": 640, "height": 480}', \
                            dest="__CAMLOWRES", \
                            required=False, \
                            type=json.loads, \
                            help="The camera low resolution (default: 640x480)")
    # Parse the command-line arguments:
    __ARGS=parser.parse_args()

    # Start the app:
    print(f"Starting {MotionDetector.version()}")
    motion_detector = MotionDetector()
    motion_detector.configMode = __ARGS.config
    motion_detector.loadConfig()

    if __ARGS.config:
        motion_detector.runConfig()
        exit(0)

    # Get a list of detected cameras:
    available_ports, working_ports, nonworking_ports = motion_detector.listCameras()
    if motion_detector.configMode or __ARGS.config or __ARGS.debug:
        print(f"Cameras:\n {working_ports}")

    if len(working_ports) > 0:
        if __ARGS.__CAMPORT >= 0:
            camera = __ARGS.__CAMPORT
        else:
            camera=working_ports[0]["port_number"]
            print(f"Using camera {working_ports[0]['port_number']} with resolution: {working_ports[0]['width']}x{working_ports[0]['height']}")
# Temporarily block out to test the config-file code.  These command line args will be able to override config-file values:
#        motion_detector.cameraPortNumber = camera
#        motion_detector.cameraRotation = __ARGS.__CAMROTATION
#        motion_detector.cameraLowResolution = __ARGS.__CAMLOWRES
#        motion_detector.cameraHighResolution = __ARGS.__CAMHIGHRES
#        motion_detector.debug = __ARGS.debug
#        motion_detector.diffingThreshold = __ARGS.__DT
#        motion_detector.minimumPixelDifference = __ARGS.__PD
#        motion_detector.showVideo = __ARGS.showvideo
#        print("Press the 'q' key to stop the app (the camera window needs to have the focus!)")

        # We're about to go in an infinite loop that will most probably run as a background process.
        # Save the processID to a file so it's easy to find and can be used to kill the process when needed.
        motion_detector.saveProcessID()
        while True:
            motion_detector.run()
            # The app may have detected that the config-file changed.  Reload in that case:
            if motion_detector._needToReload:
                print("****RELOAD****")
                motion_detector.loadConfig()
            else:
                break
        print("Ending the app")
    else:
        print("It looks like there's no camera available!")

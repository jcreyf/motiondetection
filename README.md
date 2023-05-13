# Motion Detector

This is a Python that is leveraging OpenCV to grab images from one of the cameras attached to the host and try to detect motion.<br>
The main goal for this app is to run on some small embedded devices like Raspbery PI or maybe even ESP32 and have them deployed all over the place to either alert when there is movement or to snap pictures and take video of migrating animals in my piece of the forest where I live.<br>

The app has a bunch of whistles and bells and is in POC / exploration state.  The doc will get updated when things are more concrete.<br>


## Install on Raspbery PI
https://raspberrypi-guide.github.io/programming/install-opencv

```
sudo apt-get install build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev libhdf5-103 python3-pyqt5 python3-dev -y
```

If pip opencv-python install fails:
```
sudo apt-get install python-opencv
```

https://singleboardblog.com/install-python-opencv-on-raspberry-pi/
```
pip install --upgrade opencv-contrib-python==4.7.0.72
```

https://qengineering.eu/install-opencv-4.5-on-raspberry-pi-4.html

```
/> pip install openexr
Defaulting to user installation because normal site-packages is not writeable
Looking in indexes: https://pypi.org/simple, https://www.piwheels.org/simple
Collecting openexr
Downloading https://www.piwheels.org/simple/openexr/OpenEXR-1.3.9-cp39-cp39-linux_armv7l.whl (249 kB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 249.3/249.3 kB 143.2 kB/s eta 0:00:00
Installing collected packages: openexr
Successfully installed openexr-1.3.9
```

192.168.5.220

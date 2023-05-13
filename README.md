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

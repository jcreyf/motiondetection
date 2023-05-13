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

```
/> sudo apt-get install openexr
sudo: unable to resolve host pi-cam1: Name or service not known
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
The following additional packages will be installed:
libilmbase25 libopenexr25
Suggested packages:
exrtools
The following NEW packages will be installed:
libilmbase25 libopenexr25 openexr
0 upgraded, 3 newly installed, 0 to remove and 0 not upgraded.
Need to get 976 kB of archives.
After this operation, 5,357 kB of additional disk space will be used.
Do you want to continue? [Y/n] y
Get:1 http://raspbian.mirror.axinja.net/raspbian bullseye/main armhf libilmbase25 armhf 2.5.4-1 [195 kB]
Get:2 http://raspbian.raspberrypi.org/raspbian bullseye/main armhf libopenexr25 armhf 2.5.4-2+deb11u1 [607 kB]
Get:3 http://raspbian.raspberrypi.org/raspbian bullseye/main armhf openexr armhf 2.5.4-2+deb11u1 [174 kB]
Fetched 976 kB in 3s (327 kB/s)
Selecting previously unselected package libilmbase25:armhf.
(Reading database ... 64023 files and directories currently installed.)
Preparing to unpack .../libilmbase25_2.5.4-1_armhf.deb ...
Unpacking libilmbase25:armhf (2.5.4-1) ...
Selecting previously unselected package libopenexr25:armhf.
Preparing to unpack .../libopenexr25_2.5.4-2+deb11u1_armhf.deb ...
Unpacking libopenexr25:armhf (2.5.4-2+deb11u1) ...
Selecting previously unselected package openexr.
Preparing to unpack .../openexr_2.5.4-2+deb11u1_armhf.deb ...
Unpacking openexr (2.5.4-2+deb11u1) ...
Setting up libilmbase25:armhf (2.5.4-1) ...
Setting up libopenexr25:armhf (2.5.4-2+deb11u1) ...
Setting up openexr (2.5.4-2+deb11u1) ...
Processing triggers for man-db (2.9.4-2) ...
Processing triggers for libc-bin (2.31-13+rpt2+rpi1+deb11u5) ...
```

192.168.5.220

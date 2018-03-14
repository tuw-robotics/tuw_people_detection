based on https://github.com/IntelRealSense/realsense_samples_ros

## Dependencies 

### Intel RealSense SDK for Linux
```
sudo apt-key adv --keyserver keys.gnupg.net --recv-key D6FB2970   
sudo sh -c 'echo "deb http://realsense-alm-public.s3.amazonaws.com/apt-repo xenial main" > /etc/apt/sources.list.d/realsense-latest.list'  
sudo apt update 
sudo apt install -y librealsense-object-recognition-dev librealsense-persontracking-dev librealsense-slam-dev libopencv-dev
```

## Camera Driver
```
sudo apt install ros-kinetic-realsense-camera  
```
launch file included here  
http://wiki.ros.org/realsense_camera  

## Jetson TX 2 installation instructions

NOTE: currently does not work, intel person detection binaries are not available for arm processors  

use script from https://github.com/jetsonhacks/installLibrealsenseTX2 for realsense sdk  

then install ros camera driver with  

```
sudo apt install ros-kinetic-librealsense
git clone https://github.com/intel-ros/realsense.git
cd realsense
git checkout 1.8.0
```
tuw_people_detection
===

This repository is a collection of people detection packages publishing estimates 
as `tuw_object_msgs::ObjectDetection` from [tuw_msgs](https://github.com/tuw-robotics/tuw_msgs).

# License:
Please refer to the individual packages regarding licenses.

# Included Detection Algorithms:

* ros_deep_learning (based on [ros_deep_learning](https://github.com/dusty-nv/ros_deep_learning)):
  Deep learning based people detection for cameras

* people (based on [people](https://github.com/wg-perception/people)):
  Laser based leg detector and face detector

* realsense_people_detection(based on [realsense_samples_ros](https://github.com/IntelRealSense/realsense_samples_ros)):
  Depth based people detection for Intel RealSense Cameras

* darknet_ros ([darknet_ros](https://github.com/tuw-robotics/darknet_ros)):
  ROS interface for the YOLO object detector. Provides an option for monocular detection and depth estimation through
  a TF based ground plane (at base_link) using the `rwth_ground_plane` package. If depth information from the camera
  is available the detector can use it directly (providing `monocular:=false`.

# Dependencies 

`sudo -s`
`apt install ros-melodic-bfl`
`apt install ros-melodic-kalman-filter`
`apt install ros-melodic-octomap-ros`
`apt install ros-melodic-costmap-2d`

# Dependencies for melodic

copy rwth_perception_people_msgs from spencer into this folder (TODO repo update)
https://github.com/anybotics/grid_map <- master does not compile on melodic therefore checkout fix/melodic branch!

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
  ROS interface for the YOLO object detector. The `master` branch directly uses camera depth data to obtain
  a 3D position of the detected person. The branch `devel-gp` contains a version which relies on a ground
  plane estimate to obtain a 3D position.

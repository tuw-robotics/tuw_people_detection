#ifndef REALSENSE_PEOPLE_DETECTION_NODE_H
#define REALSENSE_PEOPLE_DETECTION_NODE_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "person_tracking_video_module_factory.h"
#include "Ros2RealsenseSdkConverter.h"

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <mutex>

class RealSensePeopleDetectionNode
{
public:
  RealSensePeopleDetectionNode(ros::NodeHandle nh);
private:
  void cameraCallback(const sensor_msgs::ImageConstPtr& color_image, const sensor_msgs::ImageConstPtr& depth_image);
  void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& color_camera_info, const sensor_msgs::CameraInfoConstPtr& depth_camera_info);
  void subscribeFrameMessages();
  
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  ros::Publisher pub_person_;
  
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::CameraInfo>> sub_color_camera_info_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::CameraInfo>> sub_depth_camera_info_;
  
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> syncPolicyInfo;
  std::shared_ptr<message_filters::Synchronizer<syncPolicyInfo>> sync_camera_info_;

  std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> sub_depth_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> sub_color_;
  
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> syncPolicyImage;
  std::shared_ptr<message_filters::Synchronizer<syncPolicyImage>> sync_image_;
  
  image_transport::Publisher pub_img_;
  
  std::unique_ptr<rs::person_tracking::person_tracking_video_module_interface> pt_module_;
  std::unique_ptr<realsense_ros_person::Ros2RealSenseSdkConverter> ros2realsense_;
  
  unsigned int seq_;
  std::mutex processing_mutex_;
};

#endif // REALSENSE_PEOPLE_DETECTION_NODE_H
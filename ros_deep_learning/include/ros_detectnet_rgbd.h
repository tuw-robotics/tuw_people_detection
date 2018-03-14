#include <ros/ros.h>
#include <jetson-inference/detectNet.h>

#include <jetson-inference/loadImage.h>
#include <jetson-inference/cudaFont.h>

#include <jetson-inference/cudaMappedMemory.h>
#include <jetson-inference/cudaNormalize.h>

#include <opencv2/core.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <std_msgs/String.h>
#include <std_msgs/Int32.h>

#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>

#include <rwth_perception_people_msgs/GroundPlane.h>
#include <tuw_object_msgs/ObjectDetection.h>

#include <eigen3/Eigen/Dense>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

namespace ros_deep_learning
{
class ros_detectnet_rgbd : public nodelet::Nodelet
{
public:
  ~ros_detectnet_rgbd();
  void onInit();

private:
  void cameraCallback(const sensor_msgs::ImageConstPtr& color_image, const sensor_msgs::ImageConstPtr& depth_image,
                      const sensor_msgs::CameraInfoConstPtr& color_camera_info);

  sensor_msgs::CameraInfo color_camera_info_;
  sensor_msgs::CameraInfo depth_camera_info_;

  image_transport::Publisher impub_;

  ros::Subscriber gpsub_;
  ros::Publisher personpub_;

  std::shared_ptr<message_filters::Subscriber<sensor_msgs::CameraInfo>> sub_color_camera_info_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> sub_depth_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> sub_color_;

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image,
                                                          sensor_msgs::CameraInfo> syncPolicyImage;
  std::shared_ptr<message_filters::Synchronizer<syncPolicyImage>> sync_image_;

  detectNet* net_;

  float4* gpu_data_;

  uint32_t imgWidth_;
  uint32_t imgHeight_;
  size_t imgSize_;
};

PLUGINLIB_DECLARE_CLASS(ros_deep_learning, ros_detectnet_rgbd, ros_deep_learning::ros_detectnet_rgbd, nodelet::Nodelet);

}  // namespace ros_deep_learning

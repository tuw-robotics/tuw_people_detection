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

namespace ros_deep_learning
{
class ros_detectnet_monocular : public nodelet::Nodelet
{
public:
  ~ros_detectnet_monocular();
  void onInit();

private:
  void cameraCallback(const sensor_msgs::ImageConstPtr& input, const sensor_msgs::CameraInfoConstPtr& camera_info);

  void groundPlaneCallback(const rwth_perception_people_msgs::GroundPlane::ConstPtr& gp);

  // private variables
  // image_transport::Subscriber imsub;
  image_transport::CameraSubscriber camsub_;
  image_transport::Publisher impub_;

  ros::Subscriber gpsub_;
  ros::Publisher personpub_;

  // ros::Publisher class_pub;
  // ros::Publisher class_str_pub;
  detectNet* net_;

  float4* gpu_data_;

  uint32_t imgWidth_;
  uint32_t imgHeight_;
  size_t imgSize_;

  rwth_perception_people_msgs::GroundPlane::ConstPtr gp_;
  Eigen::Vector3f gpn_;
  float gpd_;
};

PLUGINLIB_DECLARE_CLASS(ros_deep_learning, ros_detectnet_monocular, ros_deep_learning::ros_detectnet_monocular,
                        nodelet::Nodelet);

}  // namespace ros_deep_learning

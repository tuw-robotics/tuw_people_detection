#include "realsense_people_detection/realsense_people_detection_node.h"
#include <tuw_object_msgs/ObjectDetection.h>

RealSensePeopleDetectionNode::RealSensePeopleDetectionNode(ros::NodeHandle nh) : nh_(nh), nh_private_("~"), ros2realsense_(new realsense_ros_person::Ros2RealSenseSdkConverter())
{
  image_transport::ImageTransport it(nh_);
  
  sub_color_camera_info_ = std::shared_ptr<message_filters::Subscriber<sensor_msgs::CameraInfo>>(new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, "camera_info_color", 1));
  sub_depth_camera_info_ = std::shared_ptr<message_filters::Subscriber<sensor_msgs::CameraInfo>>(new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, "camera_info_depth", 1));
  
  sync_camera_info_ = std::shared_ptr<message_filters::Synchronizer<syncPolicyInfo>>(new message_filters::Synchronizer<syncPolicyInfo>(syncPolicyInfo(40), *sub_color_camera_info_, *sub_depth_camera_info_));
  sync_camera_info_->registerCallback(boost::bind(&RealSensePeopleDetectionNode::cameraInfoCallback, this, _1, _2));
  
  pub_img_ = it.advertise("image_out", 1);
  
  pub_person_ = nh_.advertise<tuw_object_msgs::ObjectDetection>("detected_persons_tuw", 1);
  
  pt_module_.reset(rs::person_tracking::person_tracking_video_module_factory::create_person_tracking_video_module());
  //pt_module.get()->QueryConfiguration()->QueryRecognition()->Enable();
  
  ROS_INFO("constructor called");
}

void RealSensePeopleDetectionNode::cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& color_camera_info, const sensor_msgs::CameraInfoConstPtr& depth_camera_info)
{
  ROS_INFO("cameraInfoCallback");
  
  rs::core::video_module_interface::actual_module_config config = ros2realsense_->CreateSdkModuleConfig(color_camera_info, depth_camera_info);
  
  if (pt_module_ == nullptr || config.projection == nullptr)
  {
    ROS_ERROR("person tracking module or projection is null");
  }
  if (pt_module_->set_module_config(config) != rs::core::status_no_error)
  {
    ROS_ERROR("error : failed to set the enabled module configuration");
    return;
  }
  
  sub_color_camera_info_->unsubscribe();
  sub_depth_camera_info_->unsubscribe();
  
  seq_ = 0;
  subscribeFrameMessages();
}

void RealSensePeopleDetectionNode::subscribeFrameMessages()
{
  
  // Subscribe to color, point cloud and depth using synchronization filter
  ROS_INFO("subscribe to image topics");
  sub_depth_ = std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>
                     (new message_filters::Subscriber<sensor_msgs::Image>(nh_, "image_color", 1));
  sub_color_ = std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>
                     (new message_filters::Subscriber<sensor_msgs::Image>(nh_, "image_depth", 1));

  sync_image_ = std::shared_ptr<message_filters::Synchronizer<syncPolicyImage>>
                      (new message_filters::Synchronizer<syncPolicyImage>(syncPolicyImage(40), *sub_depth_, *sub_color_));
  sync_image_->registerCallback(boost::bind(&RealSensePeopleDetectionNode::cameraCallback, this, _1, _2));
  
}


void RealSensePeopleDetectionNode::cameraCallback(const sensor_msgs::ImageConstPtr& color_image, const sensor_msgs::ImageConstPtr& depth_image)
{
  //ROS_INFO("cameraCallback");
  
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(color_image, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  rs::core::correlated_sample_set sampleSet = ros2realsense_->CreateSdkSampleSet(color_image, depth_image);
  seq_++;
  {
    //prevent race condition on asynchronous  requests(change configuration/recognize) and processing
    std::lock_guard<std::mutex> guard(processing_mutex_);
    if (pt_module_->process_sample_set(sampleSet) != rs::core::status_no_error)
    {
      ROS_ERROR("error : failed to process sample");
      return;
    }
    
    // create tuw_object_msgs::ObjectDetection msg
    tuw_object_msgs::ObjectDetection detected_persons_tuw;
    detected_persons_tuw.header = depth_image->header;
    detected_persons_tuw.type = tuw_object_msgs::ObjectDetection::OBJECT_TYPE_PERSON;
    detected_persons_tuw.view_direction.w = 1;
    detected_persons_tuw.view_direction.x = 0;
    detected_persons_tuw.view_direction.y = 0;
    detected_persons_tuw.view_direction.z = 0;
    detected_persons_tuw.distance_min = 0.5;
    detected_persons_tuw.distance_max = 3.0;
    detected_persons_tuw.sensor_type = tuw_object_msgs::ObjectDetection::SENSOR_TYPE_GENERIC_RGBD;
    int number_of_detections = pt_module_->QueryOutput()->QueryNumberOfPeople();
    
    for(int index = 0; index < number_of_detections; index++)
    {
    
      Intel::RealSense::PersonTracking::PersonTrackingData::Person *personData = pt_module_->QueryOutput()->QueryPersonData(
      Intel::RealSense::PersonTracking::PersonTrackingData::AccessOrderType::ACCESS_ORDER_BY_INDEX, index);
      
      if(personData)
      {
        Intel::RealSense::PersonTracking::PersonTrackingData::PersonTracking *personTrackingData = personData->QueryTracking();
        Intel::RealSense::PersonTracking::PersonTrackingData::BoundingBox2D box = personTrackingData->Query2DBoundingBox();
        Intel::RealSense::PersonTracking::PersonTrackingData::BoundingBox2D headBoundingBox = personTrackingData->QueryHeadBoundingBox();
        Intel::RealSense::PersonTracking::PersonTrackingData::PointCombined centerMass = personTrackingData->QueryCenterMass();
        
        tuw_object_msgs::ObjectWithCovariance obj;
        
        obj.covariance_pose.emplace_back(0.2);
        obj.covariance_pose.emplace_back(0);
        obj.covariance_pose.emplace_back(0);
        obj.covariance_pose.emplace_back(0);
        obj.covariance_pose.emplace_back(0.2);
        obj.covariance_pose.emplace_back(0);
        obj.covariance_pose.emplace_back(0);
        obj.covariance_pose.emplace_back(0);
        obj.covariance_pose.emplace_back(0.2);

        obj.object.ids.emplace_back(index);
        obj.object.ids_confidence.emplace_back(1.0);
        obj.object.pose.position.x = centerMass.world.point.x;
        obj.object.pose.position.y = centerMass.world.point.y;
        obj.object.pose.position.z = centerMass.world.point.z;
        obj.object.pose.orientation.x = 0.0;
        obj.object.pose.orientation.y = 0.0;
        obj.object.pose.orientation.z = 0.0;
        obj.object.pose.orientation.w = 1.0;

        if(obj.object.pose.position.x >= 0.3 && obj.object.pose.position.x <= 3.0)
          detected_persons_tuw.objects.emplace_back(obj);
        
        // draw center of mass and bounding box
        cv::circle(cv_ptr->image, cv::Point(centerMass.image.point.x, centerMass.image.point.y), 10, CV_RGB(255,0,0), -1);
        cv::rectangle(cv_ptr->image, cv::Point(box.rect.x, box.rect.y), cv::Point(box.rect.x + box.rect.w, box.rect.y + box.rect.h), CV_RGB(0,255,0), 2, 8, 0);
      }
    }
    pub_person_.publish(detected_persons_tuw);
    pub_img_.publish(cv_ptr->toImageMsg());
  }
  ros2realsense_->ReleaseSampleSet(sampleSet);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "realsense_people_detection_node");
  ros::NodeHandle nh;
  
  RealSensePeopleDetectionNode realsense_people_detection_node(nh);
  
  //ros::spin();
  
  ros::Rate r(30);

  while (ros::ok())
  {
    ros::spinOnce();
    if (!r.sleep())
    {
      ROS_WARN("In %s: Loop missed desired rate of %.4fs (loop actually took %.4fs)", ros::this_node::getName().c_str(),
               r.expectedCycleTime().toSec(), r.cycleTime().toSec());
    }
  }
  
  return 0;
}

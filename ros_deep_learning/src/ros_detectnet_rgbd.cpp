#include "ros_detectnet_rgbd.h"

namespace ros_deep_learning
{
ros_detectnet_rgbd::~ros_detectnet_rgbd()
{
  ROS_INFO("\nshutting down...\n");
  if (gpu_data_)
    CUDA(cudaFree(gpu_data_));
  delete net_;
}
void ros_detectnet_rgbd::onInit()
{
  // get a private nodehandle
  ros::NodeHandle& private_nh = getPrivateNodeHandle();
  ros::NodeHandle& nh = getNodeHandle();

  // get parameters from server, checking for errors as it goes
  std::string prototxt_path, model_path, mean_binary_path, class_labels_path;
  if (!private_nh.getParam("prototxt_path", prototxt_path))
    ROS_ERROR("unable to read prototxt_path for detectnet node");
  if (!private_nh.getParam("model_path", model_path))
    ROS_ERROR("unable to read model_path for detectnet node");

  // make sure files exist (and we can read them)
  if (access(prototxt_path.c_str(), R_OK))
    ROS_ERROR("unable to read file \"%s\", check filename and permissions", prototxt_path.c_str());
  if (access(model_path.c_str(), R_OK))
    ROS_ERROR("unable to read file \"%s\", check filename and permissions", model_path.c_str());

  net_ = detectNet::Create(prototxt_path.c_str(), model_path.c_str(), 0.0f, 0.5f, DETECTNET_DEFAULT_INPUT,
                           DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, 2);

  if (!net_)
  {
    ROS_INFO("ros_detectnet_rgbd: failed to initialize detectNet\n");
    return;
  }

  image_transport::ImageTransport it(private_nh);

  sub_color_camera_info_ = std::shared_ptr<message_filters::Subscriber<sensor_msgs::CameraInfo>>(
      new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, "camera_info_color", 1));

  // Subscribe to color, point cloud and depth using synchronization filter
  ROS_INFO("subscribe to image topics");
  sub_depth_ = std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>(
      new message_filters::Subscriber<sensor_msgs::Image>(nh, "image_color", 1));
  sub_color_ = std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>(
      new message_filters::Subscriber<sensor_msgs::Image>(nh, "image_depth", 1));

  sync_image_ = std::shared_ptr<message_filters::Synchronizer<syncPolicyImage>>(
      new message_filters::Synchronizer<syncPolicyImage>(syncPolicyImage(40), *sub_depth_, *sub_color_,
                                                         *sub_color_camera_info_));
  sync_image_->registerCallback(boost::bind(&ros_detectnet_rgbd::cameraCallback, this, _1, _2, _3));

  impub_ = it.advertise("image_out", 1);

  personpub_ = private_nh.advertise<tuw_object_msgs::ObjectDetection>("detected_persons_tuw", 1);

  // init gpu memory
  gpu_data_ = NULL;
}

void ros_detectnet_rgbd::cameraCallback(const sensor_msgs::ImageConstPtr& color_image,
                                        const sensor_msgs::ImageConstPtr& depth_image,
                                        const sensor_msgs::CameraInfoConstPtr& color_camera_info)
{
  // color camera matrix
  Eigen::Matrix<float, 3, 3> K;
  K(0, 0) = color_camera_info->K[0];
  K(0, 1) = color_camera_info->K[1];
  K(0, 2) = color_camera_info->K[2];
  K(1, 0) = color_camera_info->K[3];
  K(1, 1) = color_camera_info->K[4];
  K(1, 2) = color_camera_info->K[5];
  K(2, 0) = color_camera_info->K[6];
  K(2, 1) = color_camera_info->K[7];
  K(2, 2) = color_camera_info->K[8];

  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K_inv = K.inverse();

  cv::Mat cv_im = cv_bridge::toCvCopy(color_image, "bgr8")->image;
  cv::Mat cv_result;

  cv::Mat cv_im_depth = cv_bridge::toCvCopy(depth_image, depth_image->encoding)->image;

  ROS_DEBUG("ros_detectnet_rgbd: image ptr at %p", cv_im.data);
  // convert bit depth
  cv_im.convertTo(cv_im, CV_32FC3);
  // convert color
  cv::cvtColor(cv_im, cv_im, CV_BGR2RGBA);

  // allocate GPU data if necessary
  if (gpu_data_ == NULL)
  {
    ROS_DEBUG("ros_detectnet_rgbd: first allocation");
    CUDA(cudaMalloc(&gpu_data_, cv_im.rows * cv_im.cols * sizeof(float4)));
  }
  else if (imgHeight_ != cv_im.rows || imgWidth_ != cv_im.cols)
  {
    ROS_DEBUG("ros_detectnet_rgbd: re allocation");
    // reallocate for a new image size if necessary
    CUDA(cudaFree(gpu_data_));
    CUDA(cudaMalloc(&gpu_data_, cv_im.rows * cv_im.cols * sizeof(float4)));
  }

  // allocate memory for output bounding boxes
  const uint32_t maxBoxes = net_->GetMaxBoundingBoxes();
  ROS_DEBUG("ros_detectnet_rgbd: maximum bounding boxes: %u\n", maxBoxes);
  const uint32_t classes = net_->GetNumClasses();

  float* bbCPU = NULL;
  float* bbCUDA = NULL;
  float* confCPU = NULL;
  float* confCUDA = NULL;

  if (!cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
      !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)))
  {
    ROS_ERROR("ros_detectnet_rgbd: failed to alloc output memory\n");
  }

  int numBoundingBoxes = maxBoxes;

  imgHeight_ = cv_im.rows;
  imgWidth_ = cv_im.cols;
  imgSize_ = cv_im.rows * cv_im.cols * sizeof(float4);
  float4* cpu_data = (float4*)(cv_im.data);

  std::vector<Eigen::Vector3f> center_points;
  std::vector<Eigen::Vector4f> bounding_box_points;

  // copy to device
  CUDA(cudaMemcpy(gpu_data_, cpu_data, imgSize_, cudaMemcpyHostToDevice));

  bool det_result = net_->Detect((float*)gpu_data_, imgWidth_, imgHeight_, bbCPU, &numBoundingBoxes, confCPU);

  if (det_result)
  {
    int lastStart = 0;

    for (int n = 0; n < numBoundingBoxes; n++)
    {
      const int nc = confCPU[n * 2 + 1];
      float* bb = bbCPU + (n * 4);

      ROS_INFO("ros_detectnet_rgbd: bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3],
               bb[2] - bb[0], bb[3] - bb[1]);

      if (!net_->DrawBoxes((float*)gpu_data_, (float*)gpu_data_, imgWidth_, imgHeight_, bbCUDA + (lastStart * 4),
                           (n - lastStart) + 1, 0))
        ROS_ERROR("ros_detectnet_rgbd: failed to draw boxes\n");

      // image coords have to be positive
      if (bb[0] > 0 && bb[1] > 0 && bb[2] > 0 && bb[3] > 0)
      {
        // calculate ground center of bounding box
        Eigen::Vector3f P1_img;
        P1_img(0) = bb[0] + (bb[2] - bb[0]) / 2;
        P1_img(1) = bb[3];
        P1_img(2) = 1;
        center_points.emplace_back(P1_img);

        bounding_box_points.emplace_back(Eigen::Vector4f(bb[0], bb[1], bb[2], bb[3]));
      }

      // copy back to host
      CUDA(cudaMemcpy(cpu_data, gpu_data_, imgSize_, cudaMemcpyDeviceToHost));

      lastStart = n;

      CUDA(cudaDeviceSynchronize());
    }
  }
  else
  {
    ROS_ERROR("ros_detectnet_rgbd: detection error occured");
  }

  Eigen::Vector3f P3D;

  cv_result = cv::Mat(imgHeight_, imgWidth_, CV_32FC4, cpu_data);
  cv_result.convertTo(cv_result, CV_8UC4);

  cv::cvtColor(cv_result, cv_result, CV_RGBA2BGR);

  tuw_object_msgs::ObjectDetection detected_persons_tuw;
  detected_persons_tuw.header = color_image->header;
  detected_persons_tuw.type = tuw_object_msgs::ObjectDetection::OBJECT_TYPE_PERSON;
  detected_persons_tuw.view_direction.w = 1;
  detected_persons_tuw.view_direction.x = 0;
  detected_persons_tuw.view_direction.y = 0;
  detected_persons_tuw.view_direction.z = 0;
  detected_persons_tuw.sensor_type = tuw_object_msgs::ObjectDetection::SENSOR_TYPE_GENERIC_RGBD;

  for (size_t i = 0; i < center_points.size(); i++)
  {
    cv::circle(cv_result, cv::Point(center_points[i](0), center_points[i](1)), 2, cv::Scalar(0, 0, 255), -1, 8, 0);

    // get 3D position from depth image bb[2] - bb[0], bb[3] - bb[1]

    // check whether roi specified by cv::Rect is inside the depth image
    // since depth fov might be smaller than rgb

    int rect_width = (int)(bounding_box_points[i](2) - bounding_box_points[i](0));
    int rect_height = (int)(bounding_box_points[i](3) - bounding_box_points[i](1));

    int rect_upper_corner_x = (int)(bounding_box_points[i](0));
    int rect_upper_corner_y = (int)(bounding_box_points[i](1));

    cv::Rect rect = cv::Rect(rect_upper_corner_x, rect_upper_corner_y, rect_width, rect_height);

    // check if bounding box is inside image
    if ((rect & cv::Rect(0, 0, cv_im_depth.cols, cv_im_depth.rows)) != rect)
      continue;

    cv::Mat bounding_box =
        cv_im_depth(cv::Rect(rect_upper_corner_x, rect_upper_corner_y, rect_width, rect_height)).clone();

    bounding_box = bounding_box.reshape(0, 1);

    std::vector<float> bounding_box_vec;
    bounding_box.copyTo(bounding_box_vec);

    std::nth_element(bounding_box_vec.begin(), bounding_box_vec.begin() + bounding_box_vec.size() / 2,
                     bounding_box_vec.end());

    double depth = double(bounding_box_vec[bounding_box_vec.size() / 2]);

    // cv::rectangle(cv_result, cv::Rect(bounding_box_points[i](0), bounding_box_points[i](1), bounding_box_points[i](2)
    // - bounding_box_points[i](0), bounding_box_points[i](3) - bounding_box_points[i](1)), cv::Scalar(0, 0, 255), -1,
    // 8, 0);

    P3D = K_inv * center_points[i];

    // check for nans
    if (!std::isnan(depth) && !std::isinf(depth) && !std::isnan(P3D(0)) && !std::isnan(P3D(1)))
    {
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

      obj.object.ids.emplace_back(i);
      obj.object.ids_confidence.emplace_back(1.0);
      obj.object.pose.position.x = P3D(0);
      obj.object.pose.position.y = P3D(1);
      obj.object.pose.position.z = depth;
      obj.object.pose.orientation.x = 0.0;
      obj.object.pose.orientation.y = 0.0;
      obj.object.pose.orientation.z = 0.0;
      obj.object.pose.orientation.w = 1.0;

      // std::cout << "position = (" << P3D(0) << ", " << P3D(1) << ", " << depth << ")" << std::endl;

      detected_persons_tuw.objects.emplace_back(obj);
    }
  }

  personpub_.publish(detected_persons_tuw);
  impub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_result).toImageMsg());
}

}  // namespace ros_deep_learning

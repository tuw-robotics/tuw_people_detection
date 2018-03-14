#include "ros_detectnet_monocular.h"

namespace ros_deep_learning
{
ros_detectnet_monocular::~ros_detectnet_monocular()
{
  ROS_INFO("\nshutting down...\n");
  if (gpu_data_)
    CUDA(cudaFree(gpu_data_));
  delete net_;
}
void ros_detectnet_monocular::onInit()
{
  // get a private nodehandle
  ros::NodeHandle& private_nh = getPrivateNodeHandle();

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
    ROS_INFO("ros_detectnet_monocular: failed to initialize detectNet\n");
    return;
  }

  gpsub_ = private_nh.subscribe("ground_plane", 100, &ros_detectnet_monocular::groundPlaneCallback, this);

  image_transport::ImageTransport it(private_nh);

  camsub_ = it.subscribeCamera("imin", 10, &ros_detectnet_monocular::cameraCallback, this);

  impub_ = it.advertise("image_out", 1);

  personpub_ = private_nh.advertise<tuw_object_msgs::ObjectDetection>("detected_persons_tuw", 1);

  // init gpu memory
  gpu_data_ = NULL;
}

void ros_detectnet_monocular::cameraCallback(const sensor_msgs::ImageConstPtr& input,
                                             const sensor_msgs::CameraInfoConstPtr& camera_info)
{
  // camera matrix
  Eigen::Matrix<float, 3, 3> K;
  K(0, 0) = camera_info->K[0];
  K(0, 1) = camera_info->K[1];
  K(0, 2) = camera_info->K[2];
  K(1, 0) = camera_info->K[3];
  K(1, 1) = camera_info->K[4];
  K(1, 2) = camera_info->K[5];
  K(2, 0) = camera_info->K[6];
  K(2, 1) = camera_info->K[7];
  K(2, 2) = camera_info->K[8];

  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K_inv = K.inverse();

  cv::Mat cv_im = cv_bridge::toCvCopy(input, "bgr8")->image;
  cv::Mat cv_result;

  ROS_DEBUG("ros_detectnet_monocular: image ptr at %p", cv_im.data);
  // convert bit depth
  cv_im.convertTo(cv_im, CV_32FC3);
  // convert color
  cv::cvtColor(cv_im, cv_im, CV_BGR2RGBA);

  // allocate GPU data if necessary
  if (gpu_data_ == NULL)
  {
    ROS_DEBUG("ros_detectnet_monocular: first allocation");
    CUDA(cudaMalloc(&gpu_data_, cv_im.rows * cv_im.cols * sizeof(float4)));
  }
  else if (imgHeight_ != cv_im.rows || imgWidth_ != cv_im.cols)
  {
    ROS_DEBUG("ros_detectnet_monocular: re allocation");
    // reallocate for a new image size if necessary
    CUDA(cudaFree(gpu_data_));
    CUDA(cudaMalloc(&gpu_data_, cv_im.rows * cv_im.cols * sizeof(float4)));
  }

  // allocate memory for output bounding boxes
  const uint32_t maxBoxes = net_->GetMaxBoundingBoxes();
  ROS_DEBUG("ros_detectnet_monocular: maximum bounding boxes: %u\n", maxBoxes);
  const uint32_t classes = net_->GetNumClasses();

  float* bbCPU = NULL;
  float* bbCUDA = NULL;
  float* confCPU = NULL;
  float* confCUDA = NULL;

  if (!cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
      !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)))
  {
    ROS_ERROR("ros_detectnet_monocular: failed to alloc output memory\n");
  }

  int numBoundingBoxes = maxBoxes;

  imgHeight_ = cv_im.rows;
  imgWidth_ = cv_im.cols;
  imgSize_ = cv_im.rows * cv_im.cols * sizeof(float4);
  float4* cpu_data = (float4*)(cv_im.data);

  std::vector<Eigen::Vector3f> center_points;
  std::vector<double> bb_heights;

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

      ROS_INFO("ros_detectnet_monocular: bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2],
               bb[3], bb[2] - bb[0], bb[3] - bb[1]);

      if (!net_->DrawBoxes((float*)gpu_data_, (float*)gpu_data_, imgWidth_, imgHeight_, bbCUDA + (lastStart * 4),
                           (n - lastStart) + 1, 0))
        ROS_ERROR("ros_detectnet_monocular: failed to draw boxes\n");

      // calculate ground center of bounding box
      Eigen::Vector3f P1_img;
      P1_img(0) = bb[0] + (bb[2] - bb[0]) / 2;
      P1_img(1) = bb[3];
      P1_img(2) = 1;
      center_points.emplace_back(P1_img);
      bb_heights.emplace_back(bb[3] - bb[1]);

      // copy back to host
      CUDA(cudaMemcpy(cpu_data, gpu_data_, imgSize_, cudaMemcpyDeviceToHost));

      lastStart = n;

      CUDA(cudaDeviceSynchronize());
    }
  }
  else
  {
    ROS_ERROR("ros_detectnet_monocular: detection error occured");
  }

  Eigen::Vector3f P0(0, 0, 0);
  Eigen::Vector3f P1;
  Eigen::Vector3f P1_ground;
  Eigen::Vector3f P_diff;
  Eigen::Vector3f P3D;

  Eigen::Vector2f eigenvector1;
  Eigen::Vector2f eigenvector2;
  double eigenvalue1 = 800;  // 50
  double eigenvalue2 = 0.02;

  cv_result = cv::Mat(imgHeight_, imgWidth_, CV_32FC4, cpu_data);
  cv_result.convertTo(cv_result, CV_8UC4);

  cv::cvtColor(cv_result, cv_result, CV_RGBA2BGR);

  tuw_object_msgs::ObjectDetection detected_persons_tuw;
  detected_persons_tuw.header = input->header;
  detected_persons_tuw.type = tuw_object_msgs::ObjectDetection::OBJECT_TYPE_PERSON;
  detected_persons_tuw.view_direction.w = 1;
  detected_persons_tuw.view_direction.x = 0;
  detected_persons_tuw.view_direction.y = 0;
  detected_persons_tuw.view_direction.z = 0;
  detected_persons_tuw.sensor_type = tuw_object_msgs::ObjectDetection::SENSOR_TYPE_GENERIC_MONOCULAR_VISION;

  for (size_t i = 0; i < center_points.size(); i++)
  {
    cv::circle(cv_result, cv::Point(center_points[i](0), center_points[i](1)), 2, cv::Scalar(0, 0, 255), -1, 8, 0);

    // calculate 3D position through intersection with ground plane
    P1 = K_inv * center_points[i];
    P_diff = P1 - P0;
    float nom = gpd_ - gpn_.dot(P0);
    float denom = gpn_.dot(P_diff);

    if (denom != 0)
    {
      P3D = P0 + nom / denom * P_diff;

      // move point P1 onto the ground plane s.t.
      // P1_ground, P3D define a line segement on the GP in direction towards the detection
      P1_ground = P1;
      P1_ground(1) = P3D(1);  // y is coordinate to ground

      eigenvector1(0) = (P3D - P1_ground).normalized()(0);
      eigenvector1(1) = (P3D - P1_ground).normalized()(2);
      // second eigenvector is orthogonal to first
      // i.e. eigenvector1 . eigenvector2 = 0
      eigenvector2(0) = -eigenvector1(1);
      eigenvector2(1) = eigenvector1(0);

      // construct covariance from eigenvectors

      Eigen::Matrix2d P;
      Eigen::Matrix<double, 2, 2, Eigen::RowMajor> Q;
      Eigen::Matrix<double, 2, 2> diag;
      diag << eigenvalue1, 0, 0, eigenvalue2;

      P.leftCols(1) = eigenvector1.cast<double>();
      P.rightCols(1) = eigenvector2.cast<double>();

      Q = P * diag * P.inverse();

      tuw_object_msgs::ObjectWithCovariance obj;

      // points defining the direction towards the detection
      obj.object.shape_variables.emplace_back(P1_ground(0));
      obj.object.shape_variables.emplace_back(P1_ground(1));
      obj.object.shape_variables.emplace_back(P1_ground(2));

      obj.object.shape_variables.emplace_back(P3D(0));
      obj.object.shape_variables.emplace_back(P3D(1));
      obj.object.shape_variables.emplace_back(P3D(2));

      obj.covariance_pose.emplace_back(Q(0, 0));
      obj.covariance_pose.emplace_back(0);
      obj.covariance_pose.emplace_back(Q(1, 0));

      obj.covariance_pose.emplace_back(0);
      obj.covariance_pose.emplace_back(0);
      obj.covariance_pose.emplace_back(0);

      obj.covariance_pose.emplace_back(Q(0, 1));
      obj.covariance_pose.emplace_back(0);
      obj.covariance_pose.emplace_back(Q(1, 1));

      obj.object.ids.emplace_back(i);
      obj.object.ids_confidence.emplace_back(1.0);
      obj.object.pose.position.x = P3D(0);
      obj.object.pose.position.y = P3D(1);
      obj.object.pose.position.z = P3D(2);
      obj.object.pose.orientation.x = 0.0;
      obj.object.pose.orientation.y = 0.0;
      obj.object.pose.orientation.z = 0.0;
      obj.object.pose.orientation.w = 1.0;

      // filter inaccurate detections
      if (std::hypot(P3D(0), P3D(2)) <= 9.0 && P3D(2) >= 0.0)
      {
        detected_persons_tuw.objects.emplace_back(obj);
      }
    }
  }

  personpub_.publish(detected_persons_tuw);
  impub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_result).toImageMsg());
}

void ros_detectnet_monocular::groundPlaneCallback(const rwth_perception_people_msgs::GroundPlane::ConstPtr& gp)
{
  gp_ = gp;

  // ground plane normal vector
  gpn_(0) = gp_->n[0];
  gpn_(1) = gp_->n[1];
  gpn_(2) = gp_->n[2];

  // ground plane distance
  gpd_ = ((float)gp_->d) * (-1.0);
}

}  // namespace ros_deep_learning

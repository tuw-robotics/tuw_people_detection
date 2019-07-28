/*
 * YoloObjectDetector.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

// yolo object detector
#include "darknet_ros/YoloObjectDetector.hpp"

// Check for xServer
#include <X11/Xlib.h>

#ifdef DARKNET_FILE_PATH
std::string darknetFilePath_ = DARKNET_FILE_PATH;
#else
#error Path of darknet repository is not defined in CMakeLists.txt.
#endif

namespace darknet_ros {

char *cfg;
char *weights;
char *data;
char **detectionNames;

YoloObjectDetector::YoloObjectDetector(ros::NodeHandle nh)
    : nodeHandle_(nh),
      imageTransport_(nodeHandle_),
      numClasses_(0),
      classLabels_(0),
      rosBoxes_(0),
      rosBoxCounter_(0)
{
  ROS_INFO("[YoloObjectDetector] Node started.");

  // Read parameters from config file.
  if (!readParameters()) {
    ros::requestShutdown();
  }

  init();
}

YoloObjectDetector::~YoloObjectDetector()
{
  {
    boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
    isNodeRunning_ = false;
  }
  yoloThread_.join();
}

bool YoloObjectDetector::readParameters()
{
  // Load common parameters.
  nodeHandle_.param("image_view/enable_opencv", viewImage_, true);
  nodeHandle_.param("image_view/wait_key_delay", waitKeyDelay_, 3);
  nodeHandle_.param("image_view/enable_console_output", enableConsoleOutput_, false);

  // Check if Xserver is running on Linux.
  if (XOpenDisplay(NULL)) {
    // Do nothing!
    ROS_INFO("[YoloObjectDetector] Xserver is running.");
  } else {
    ROS_INFO("[YoloObjectDetector] Xserver is not running.");
    viewImage_ = false;
  }

  // Set vector sizes.
  nodeHandle_.param("yolo_model/detection_classes/names", classLabels_,
                    std::vector<std::string>(0));
  numClasses_ = classLabels_.size();
  rosBoxes_ = std::vector<std::vector<RosBox_> >(numClasses_);
  rosBoxCounter_ = std::vector<int>(numClasses_);

  return true;
}

void YoloObjectDetector::init()
{
  ROS_INFO("[YoloObjectDetector] init().");

  // Initialize deep network of darknet.
  std::string weightsPath;
  std::string configPath;
  std::string dataPath;
  std::string configModel;
  std::string weightsModel;

  nodeHandle_.param("monocular", monocular_, false);

  // Threshold of object detection.
  float thresh;
  nodeHandle_.param("yolo_model/threshold/value", thresh, (float) 0.3);

  // Path to weights file.
  nodeHandle_.param("yolo_model/weight_file/name", weightsModel,
                    std::string("yolov2-tiny.weights"));
  nodeHandle_.param("weights_path", weightsPath, std::string("/default"));
  weightsPath += "/" + weightsModel;
  weights = new char[weightsPath.length() + 1];
  strcpy(weights, weightsPath.c_str());

  // Path to config file.
  nodeHandle_.param("yolo_model/config_file/name", configModel, std::string("yolov2-tiny.cfg"));
  nodeHandle_.param("config_path", configPath, std::string("/default"));
  configPath += "/" + configModel;
  cfg = new char[configPath.length() + 1];
  strcpy(cfg, configPath.c_str());

  // Path to data folder.
  dataPath = darknetFilePath_;
  dataPath += "/data";
  data = new char[dataPath.length() + 1];
  strcpy(data, dataPath.c_str());

  // Get classes.
  detectionNames = (char**) realloc((void*) detectionNames, (numClasses_ + 1) * sizeof(char*));
  for (int i = 0; i < numClasses_; i++) {
    detectionNames[i] = new char[classLabels_[i].length() + 1];
    strcpy(detectionNames[i], classLabels_[i].c_str());
  }

  // Load network.
  setupNetwork(cfg, weights, data, thresh, detectionNames, numClasses_,
                0, 0, 1, 0.5, 0, 0, 0, 0);
  yoloThread_ = std::thread(&YoloObjectDetector::yolo, this);

  // Initialize publisher and subscriber.
  std::string cameraTopicName;
  int cameraQueueSize;
  std::string objectDetectorTopicName;
  int objectDetectorQueueSize;
  bool objectDetectorLatch;
  std::string boundingBoxesTopicName;
  int boundingBoxesQueueSize;
  bool boundingBoxesLatch;
  std::string detectionImageTopicName;
  int detectionImageQueueSize;
  bool detectionImageLatch;

  nodeHandle_.param("subscribers/camera_reading/topic", cameraTopicName,
                    std::string("/camera/image_raw"));
  nodeHandle_.param("subscribers/camera_reading/queue_size", cameraQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/topic", objectDetectorTopicName,
                    std::string("found_object"));
  nodeHandle_.param("publishers/object_detector/queue_size", objectDetectorQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/latch", objectDetectorLatch, false);
  nodeHandle_.param("publishers/bounding_boxes/topic", boundingBoxesTopicName,
                    std::string("bounding_boxes"));
  nodeHandle_.param("publishers/bounding_boxes/queue_size", boundingBoxesQueueSize, 1);
  nodeHandle_.param("publishers/bounding_boxes/latch", boundingBoxesLatch, false);
  nodeHandle_.param("publishers/detection_image/topic", detectionImageTopicName,
                    std::string("detection_image"));
  nodeHandle_.param("publishers/detection_image/queue_size", detectionImageQueueSize, 1);
  nodeHandle_.param("publishers/detection_image/latch", detectionImageLatch, true);
  
  if(monocular_)
    gpSubscriber_ = nodeHandle_.subscribe("ground_plane", 100, &YoloObjectDetector::groundPlaneCallback, this);

  //imageSubscriber_ = imageTransport_.subscribe(cameraTopicName, cameraQueueSize,
  //                                             &YoloObjectDetector::cameraCallback, this);
  objectPublisher_ = nodeHandle_.advertise<std_msgs::Int8>(objectDetectorTopicName,
                                                           objectDetectorQueueSize,
                                                           objectDetectorLatch);

  objectDetectionPublisher_ = nodeHandle_.advertise<tuw_object_msgs::ObjectDetection>("object_detections_tuw", 1);

  boundingBoxesPublisher_ = nodeHandle_.advertise<darknet_ros_msgs::BoundingBoxes>(
      boundingBoxesTopicName, boundingBoxesQueueSize, boundingBoxesLatch);
  detectionImagePublisher_ = nodeHandle_.advertise<sensor_msgs::Image>(detectionImageTopicName,
                                                                       detectionImageQueueSize,
                                                                       detectionImageLatch);
cameraInfoSubscriber_ = std::unique_ptr<message_filters::Subscriber<sensor_msgs::CameraInfo>>(
      new message_filters::Subscriber<sensor_msgs::CameraInfo>(nodeHandle_, "camera_info", 1));

  depthImageSubscriber_ = std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image>>(
      new message_filters::Subscriber<sensor_msgs::Image>(nodeHandle_, "image_depth", 1));

  colorImageSubscriber_ = std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image>>(
      new message_filters::Subscriber<sensor_msgs::Image>(nodeHandle_, "image_color", 1));

  if (monocular_)
  {
    sensor_msgs::ImageConstPtr dummy_msg(new sensor_msgs::Image());
    syncImageMonocular_ = std::unique_ptr<message_filters::Synchronizer<syncPolicyImageMonocular>>(
        new message_filters::Synchronizer<syncPolicyImageMonocular>(syncPolicyImageMonocular(40),
                                                                    *colorImageSubscriber_, *cameraInfoSubscriber_));

    syncImageMonocular_->registerCallback(boost::bind(&YoloObjectDetector::cameraCallback, this, _1, dummy_msg, _2));
  }
  else
  {
    syncImage_ = std::unique_ptr<message_filters::Synchronizer<syncPolicyImage>>(
        new message_filters::Synchronizer<syncPolicyImage>(syncPolicyImage(40), *colorImageSubscriber_,
                                                           *depthImageSubscriber_, *cameraInfoSubscriber_));
    syncImage_->registerCallback(boost::bind(&YoloObjectDetector::cameraCallback, this, _1, _2, _3));
  }

  // Action servers.
  std::string checkForObjectsActionName;
  nodeHandle_.param("actions/camera_reading/topic", checkForObjectsActionName,
                    std::string("check_for_objects"));
  checkForObjectsActionServer_.reset(
      new CheckForObjectsActionServer(nodeHandle_, checkForObjectsActionName, false));
  checkForObjectsActionServer_->registerGoalCallback(
      boost::bind(&YoloObjectDetector::checkForObjectsActionGoalCB, this));
  checkForObjectsActionServer_->registerPreemptCallback(
      boost::bind(&YoloObjectDetector::checkForObjectsActionPreemptCB, this));
  checkForObjectsActionServer_->start();
}

// void YoloObjectDetector::cameraCallback(const sensor_msgs::ImageConstPtr& msg)
void YoloObjectDetector::cameraCallback(const sensor_msgs::ImageConstPtr &color_image,
                                        const sensor_msgs::ImageConstPtr &depth_image,
                                        const sensor_msgs::CameraInfoConstPtr &camera_info)
{
  ROS_DEBUG("[YoloObjectDetector] USB image received.");

  cv_bridge::CvImagePtr cam_image;
  cv_bridge::CvImagePtr cam_image_depth;

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

  K_inv_ = K.inverse();

  try
  {
    cam_image = cv_bridge::toCvCopy(color_image, sensor_msgs::image_encodings::BGR8);
    imageHeader_ = color_image->header;
  }
  catch (cv_bridge::Exception &e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (!monocular_)
  {
    try
    {
      cam_image_depth = cv_bridge::toCvCopy(depth_image, depth_image->encoding);
      //cam_mat_depth = cv::Mat::zeros(depth_image->height, depth_image->width, CV_32F);
      //std::cout << "setting mat " << depth_image->encoding << std::endl;
      //std::cout << "depth width " << depth_image->width << std::endl;
      //std::cout << "depth height " << depth_image->height << std::endl;
      //std::cout << "is big endian " << depth_image->is_bigendian << std::endl;
      //std::cout << depth_image->data.size() << std::endl;

      //for (int r = 0; r < depth_image->width; ++r)
      //{
      //  for (int c = 0; c < depth_image->height * 2; c += 2)
      //  {
      //    cam_mat_depth.at<float>(c,r) = static_cast<float>((uint16_t) depth_image->data[r * depth_image->step + c] << 8 | depth_image->data[r * depth_image->step + c + 1]);// / 16.0f ;
      //  }
      //}

    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
  }

  if (cam_image) {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      imageHeader_ = cam_image->header;
      camImageCopy_ = cam_image->image.clone();
      if (cam_image_depth)
        cam_image_depth->image.convertTo(camImageDepthCopy_, CV_64F, 1.0/1000.0);
        //camImageDepthCopy_ = cam_image_depth->image;
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

void YoloObjectDetector::checkForObjectsActionGoalCB()
{
  ROS_DEBUG("[YoloObjectDetector] Start check for objects action.");

  boost::shared_ptr<const darknet_ros_msgs::CheckForObjectsGoal> imageActionPtr =
      checkForObjectsActionServer_->acceptNewGoal();
  sensor_msgs::Image imageAction = imageActionPtr->image;

  cv_bridge::CvImagePtr cam_image;

  try {
    cam_image = cv_bridge::toCvCopy(imageAction, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image) {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      camImageCopy_ = cam_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexActionStatus_);
      actionId_ = imageActionPtr->id;
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

void YoloObjectDetector::checkForObjectsActionPreemptCB()
{
  ROS_DEBUG("[YoloObjectDetector] Preempt check for objects action.");
  checkForObjectsActionServer_->setPreempted();
}

bool YoloObjectDetector::isCheckingForObjects() const
{
  return (ros::ok() && checkForObjectsActionServer_->isActive()
      && !checkForObjectsActionServer_->isPreemptRequested());
}

bool YoloObjectDetector::publishDetectionImage(const cv::Mat& detectionImage)
{
  if (detectionImagePublisher_.getNumSubscribers() < 1)
    return false;
  cv_bridge::CvImage cvImage;
  cvImage.header.stamp = ros::Time::now();
  cvImage.header.frame_id = "detection_image";
  cvImage.encoding = sensor_msgs::image_encodings::BGR8;
  cvImage.image = detectionImage;
  detectionImagePublisher_.publish(*cvImage.toImageMsg());
  ROS_DEBUG("Detection image has been published.");
  return true;
}

// double YoloObjectDetector::getWallTime()
// {
//   struct timeval time;
//   if (gettimeofday(&time, NULL)) {
//     return 0;
//   }
//   return (double) time.tv_sec + (double) time.tv_usec * .000001;
// }

int YoloObjectDetector::sizeNetwork(network *net)
{
  int i;
  int count = 0;
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      count += l.outputs;
    }
  }
  return count;
}

void YoloObjectDetector::rememberNetwork(network *net)
{
  int i;
  int count = 0;
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      memcpy(predictions_[demoIndex_] + count, net->layers[i].output, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
}

detection *YoloObjectDetector::avgPredictions(network *net, int *nboxes)
{
  int i, j;
  int count = 0;
  fill_cpu(demoTotal_, 0, avg_, 1);
  for(j = 0; j < demoFrame_; ++j){
    axpy_cpu(demoTotal_, 1./demoFrame_, predictions_[j], 1, avg_, 1);
  }
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      memcpy(l.output, avg_ + count, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
  detection *dets = get_network_boxes(net, buff_[0].w, buff_[0].h, demoThresh_, demoHier_, 0, 1, nboxes);
  return dets;
}

void *YoloObjectDetector::detectInThread()
{
  running_ = 1;
  float nms = .4;

  layer l = net_->layers[net_->n - 1];
  float *X = buffLetter_[(buffIndex_ + 2) % 3].data;
  float *prediction = network_predict(net_, X);

  rememberNetwork(net_);
  detection *dets = 0;
  int nboxes = 0;
  dets = avgPredictions(net_, &nboxes);

  if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

  if (enableConsoleOutput_) {
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps_);
    printf("Objects:\n\n");
  }
  image display = buff_[(buffIndex_+2) % 3];
  draw_detections(display, dets, nboxes, demoThresh_, demoNames_, demoAlphabet_, demoClasses_);

  // extract the bounding boxes and send them to ROS
  int i, j;
  int count = 0;
  for (i = 0; i < nboxes; ++i) {
    float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
    float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
    float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
    float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

    if (xmin < 0)
      xmin = 0;
    if (ymin < 0)
      ymin = 0;
    if (xmax > 1)
      xmax = 1;
    if (ymax > 1)
      ymax = 1;

    // iterate through possible boxes and collect the bounding boxes
    for (j = 0; j < demoClasses_; ++j) {
      if (dets[i].prob[j]) {
        float x_center = (xmin + xmax) / 2;
        float y_center = (ymin + ymax) / 2;
        float BoundingBox_width = xmax - xmin;
        float BoundingBox_height = ymax - ymin;

        // define bounding box
        // BoundingBox must be 1% size of frame (3.2x2.4 pixels)
        if (BoundingBox_width > 0.01 && BoundingBox_height > 0.01) {
          roiBoxes_[count].x = x_center;
          roiBoxes_[count].y = y_center;
          roiBoxes_[count].w = BoundingBox_width;
          roiBoxes_[count].h = BoundingBox_height;
          roiBoxes_[count].Class = j;
          roiBoxes_[count].prob = dets[i].prob[j];
          count++;
        }
      }
    }
  }

  // create array to store found bounding boxes
  // if no object detected, make sure that ROS knows that num = 0
  if (count == 0) {
    roiBoxes_[0].num = 0;
  } else {
    roiBoxes_[0].num = count;
  }

  free_detections(dets, nboxes);
  demoIndex_ = (demoIndex_ + 1) % demoFrame_;
  running_ = 0;
  return 0;
}

void *YoloObjectDetector::fetchInThread()
{
  IplImageWithHeader_ imageAndHeader = getIplImageWithHeader();
  IplImage* ROS_img = imageAndHeader.image;
  ipl_into_image(ROS_img, buff_[buffIndex_]);
  headerBuff_[buffIndex_] = imageAndHeader.header;
  {
    boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
    buffId_[buffIndex_] = actionId_;
  }
  rgbgr_image(buff_[buffIndex_]);
  letterbox_image_into(buff_[buffIndex_], net_->w, net_->h, buffLetter_[buffIndex_]);
  return 0;
}

void *YoloObjectDetector::displayInThread(void *ptr)
{
  show_image_cv(buff_[(buffIndex_ + 1)%3], "YOLO V3", ipl_);
  int c = cvWaitKey(waitKeyDelay_);
  if (c != -1) c = c%256;
  if (c == 27) {
      demoDone_ = 1;
      return 0;
  } else if (c == 82) {
      demoThresh_ += .02;
  } else if (c == 84) {
      demoThresh_ -= .02;
      if(demoThresh_ <= .02) demoThresh_ = .02;
  } else if (c == 83) {
      demoHier_ += .02;
  } else if (c == 81) {
      demoHier_ -= .02;
      if(demoHier_ <= .0) demoHier_ = .0;
  }
  return 0;
}

void *YoloObjectDetector::displayLoop(void *ptr)
{
  while (1) {
    displayInThread(0);
  }
}

void *YoloObjectDetector::detectLoop(void *ptr)
{
  while (1) {
    detectInThread();
  }
}

void YoloObjectDetector::setupNetwork(char *cfgfile, char *weightfile, char *datafile, float thresh,
                                      char **names, int classes,
                                      int delay, char *prefix, int avg_frames, float hier, int w, int h,
                                      int frames, int fullscreen)
{
  demoPrefix_ = prefix;
  demoDelay_ = delay;
  demoFrame_ = avg_frames;
  image **alphabet = load_alphabet_with_file(datafile);
  demoNames_ = names;
  demoAlphabet_ = alphabet;
  demoClasses_ = classes;
  demoThresh_ = thresh;
  demoHier_ = hier;
  fullScreen_ = fullscreen;
  printf("YOLO V3\n");
  net_ = load_network(cfgfile, weightfile, 0);
  set_batch_network(net_, 1);
}

void YoloObjectDetector::yolo()
{
  const auto wait_duration = std::chrono::milliseconds(2000);
  while (!getImageStatus()) {
    printf("Waiting for image.\n");
    if (!isNodeRunning()) {
      return;
    }
    std::this_thread::sleep_for(wait_duration);
  }

  std::thread detect_thread;
  std::thread fetch_thread;

  srand(2222222);

  int i;
  demoTotal_ = sizeNetwork(net_);
  predictions_ = (float **) calloc(demoFrame_, sizeof(float*));
  for (i = 0; i < demoFrame_; ++i){
      predictions_[i] = (float *) calloc(demoTotal_, sizeof(float));
  }
  avg_ = (float *) calloc(demoTotal_, sizeof(float));

  layer l = net_->layers[net_->n - 1];
  roiBoxes_ = (darknet_ros::RosBox_ *) calloc(l.w * l.h * l.n, sizeof(darknet_ros::RosBox_));

  IplImageWithHeader_ imageAndHeader = getIplImageWithHeader();
  IplImage* ROS_img = imageAndHeader.image;
  buff_[0] = ipl_to_image(ROS_img);
  buff_[1] = copy_image(buff_[0]);
  buff_[2] = copy_image(buff_[0]);
  headerBuff_[0] = imageAndHeader.header;
  headerBuff_[1] = headerBuff_[0];
  headerBuff_[2] = headerBuff_[0];
  buffLetter_[0] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[1] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[2] = letterbox_image(buff_[0], net_->w, net_->h);
  ipl_ = cvCreateImage(cvSize(buff_[0].w, buff_[0].h), IPL_DEPTH_8U, buff_[0].c);

  int count = 0;

  if (!demoPrefix_ && viewImage_) {
    cvNamedWindow("YOLO V3", CV_WINDOW_NORMAL);
    if (fullScreen_) {
      cvSetWindowProperty("YOLO V3", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    } else {
      cvMoveWindow("YOLO V3", 0, 0);
      cvResizeWindow("YOLO V3", 640, 480);
    }
  }

  demoTime_ = what_time_is_it_now();

  while (!demoDone_) {
    buffIndex_ = (buffIndex_ + 1) % 3;
    fetch_thread = std::thread(&YoloObjectDetector::fetchInThread, this);
    detect_thread = std::thread(&YoloObjectDetector::detectInThread, this);
    if (!demoPrefix_) {
      fps_ = 1./(what_time_is_it_now() - demoTime_);
      demoTime_ = what_time_is_it_now();
      if (viewImage_) {
        displayInThread(0);
      }
      publishInThread();
    } else {
      char name[256];
      sprintf(name, "%s_%08d", demoPrefix_, count);
      save_image(buff_[(buffIndex_ + 1) % 3], name);
    }
    fetch_thread.join();
    detect_thread.join();
    ++count;
    if (!isNodeRunning()) {
      demoDone_ = true;
    }
  }

}

IplImageWithHeader_ YoloObjectDetector::getIplImageWithHeader()
{
  boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
  IplImage* ROS_img = new IplImage(camImageCopy_);
  IplImageWithHeader_ header = {.image = ROS_img, .header = imageHeader_};
  return header;
}

bool YoloObjectDetector::getImageStatus(void)
{
  boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
  return imageStatus_;
}

bool YoloObjectDetector::isNodeRunning(void)
{
  boost::shared_lock<boost::shared_mutex> lock(mutexNodeStatus_);
  return isNodeRunning_;
}

void *YoloObjectDetector::publishInThread()
{
  // Publish image.
  cv::Mat cvImage = cv::cvarrToMat(ipl_);
  if (!publishDetectionImage(cv::Mat(cvImage))) {
    ROS_DEBUG("Detection image has not been broadcasted.");
  }

  tuw_object_msgs::ObjectDetection detected_persons_tuw;
  detected_persons_tuw.header = imageHeader_;
  detected_persons_tuw.type = tuw_object_msgs::ObjectDetection::OBJECT_TYPE_PERSON;
  detected_persons_tuw.view_direction.w = 1;
  detected_persons_tuw.view_direction.x = 0;
  detected_persons_tuw.view_direction.y = 0;
  detected_persons_tuw.view_direction.z = 0;
  detected_persons_tuw.sensor_type = tuw_object_msgs::ObjectDetection::SENSOR_TYPE_GENERIC_RGBD;

  Eigen::Vector3f P3D;
  Eigen::Vector3f center_point;
  Eigen::Vector3f P0(0, 0, 0);
  Eigen::Vector3f P1;
  Eigen::Vector3f P1_ground;
  Eigen::Vector3f P_diff;

  Eigen::Vector2f eigenvector1;
  Eigen::Vector2f eigenvector2;
  double eigenvalue1 = 10;  // 50
  double eigenvalue2 = 0.1;

  // Publish bounding boxes and detection result.
  int num = roiBoxes_[0].num;
  if (num > 0 && num <= 100) {
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < numClasses_; j++) {
        if (roiBoxes_[i].Class == j) {
          rosBoxes_[j].push_back(roiBoxes_[i]);
          rosBoxCounter_[j]++;
        }
      }
    }

    std_msgs::Int8 msg;
    msg.data = num;
    objectPublisher_.publish(msg);

    for (int i = 0; i < numClasses_; i++) {
      if (rosBoxCounter_[i] > 0) {
        darknet_ros_msgs::BoundingBox boundingBox;

        for (int j = 0; j < rosBoxCounter_[i]; j++) {
          int xmin = (rosBoxes_[i][j].x - rosBoxes_[i][j].w / 2) * frameWidth_;
          int ymin = (rosBoxes_[i][j].y - rosBoxes_[i][j].h / 2) * frameHeight_;
          int xmax = (rosBoxes_[i][j].x + rosBoxes_[i][j].w / 2) * frameWidth_;
          int ymax = (rosBoxes_[i][j].y + rosBoxes_[i][j].h / 2) * frameHeight_;

          boundingBox.Class = classLabels_[i];
          boundingBox.probability = rosBoxes_[i][j].prob;
          boundingBox.xmin = xmin;
          boundingBox.ymin = ymin;
          boundingBox.xmax = xmax;
          boundingBox.ymax = ymax;
          boundingBoxesResults_.bounding_boxes.push_back(boundingBox);

          if (boundingBox.Class == std::string("person"))
          {
            center_point(0) = xmin + (xmax - xmin) / 2;
            center_point(1) = ymax;
            center_point(2) = 1;

            cv::circle(camImageCopy_, cv::Point(center_point(0), center_point(1)), 4, cv::Scalar(0, 0, 255), -1, 8, 0);

            if (monocular_)
            {
              std::cout << "monocular detection " << std::endl;
              // calculate 3D position through intersection with ground plane
              P1 = K_inv_ * center_point;
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

                // check for nans
                if (!std::isnan(P3D(0)) && !std::isnan(P3D(1)))
                {
                  tuw_object_msgs::ObjectWithCovariance obj;

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

                  if (std::hypot(P3D(0), P3D(2)) <= 9.0 && P3D(2) >= 0.0)
                  {
                    detected_persons_tuw.objects.emplace_back(obj);
                  }
                }
              }
            }
            else
            {
                std::cout << "stereo detection" << std::endl;
              int rect_width = (int)(xmax - xmin);
              int rect_height = (int)(ymax - ymin);

              cv::Rect rect = cv::Rect(xmin, ymin, rect_width, rect_height);
              // cv::rectangle(camImageCopy_, rect, cv::Scalar(0, 0, 255), 3, 8, 0);

              if ((rect & cv::Rect(0, 0, camImageDepthCopy_.cols, camImageDepthCopy_.rows)) != rect)
              {
                  std::cout << "cant copy depth image " << std::endl;
                  continue;
              }

              cv::Mat bb = camImageDepthCopy_(rect).clone();

              bb = bb.reshape(0, 1);

              std::vector<float> bb_vec;
              bb.copyTo(bb_vec);

              bb_vec.erase(std::remove_if(bb_vec.begin(), bb_vec.end(),
                                          [](float &x)
                                          {
                                            return (std::isnan(x) || std::isinf(x));
                                          }),
                           bb_vec.end());

              double depth = 0;

              std::cout << "bb vec size: " << bb_vec.size() << std::endl;

              // median depth
              if (bb_vec.size() % 2 == 0)
              {
                const auto median_it1 = bb_vec.begin() + bb_vec.size() / 2 - 1;
                const auto median_it2 = bb_vec.begin() + bb_vec.size() / 2;

                std::nth_element(bb_vec.begin(), median_it1, bb_vec.end());
                const double median_1 = *median_it1;
                std::nth_element(bb_vec.begin(), median_it2, bb_vec.end());
                const double median_2 = *median_it2;

                depth = (median_1 + median_2) / 2;
              }
              else
              {
                const auto median_it = bb_vec.begin() + bb_vec.size() / 2;
                std::nth_element(bb_vec.begin(), median_it, bb_vec.end());
                depth = *median_it;
              }

              P3D = K_inv_ * center_point;

              std::cout << "depth " << depth << std::endl;
              std::cout << "P3D 0 " << P3D(0) << std::endl;
              std::cout << "P3D 1 " << P3D(1) << std::endl;

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

                detected_persons_tuw.objects.emplace_back(obj);
              }
            }
          }
        }
      }
    }
    boundingBoxesResults_.header.stamp = ros::Time::now();
    boundingBoxesResults_.header.frame_id = "detection";
    boundingBoxesResults_.image_header = headerBuff_[(buffIndex_ + 1) % 3];
    boundingBoxesPublisher_.publish(boundingBoxesResults_);
  } else {
    std_msgs::Int8 msg;
    msg.data = 0;
    objectPublisher_.publish(msg);
  }
  if (isCheckingForObjects()) {
    ROS_DEBUG("[YoloObjectDetector] check for objects in image.");
    darknet_ros_msgs::CheckForObjectsResult objectsActionResult;
    objectsActionResult.id = buffId_[0];
    objectsActionResult.bounding_boxes = boundingBoxesResults_;
    checkForObjectsActionServer_->setSucceeded(objectsActionResult, "Send bounding boxes.");
  }
  if (detected_persons_tuw.objects.size() > 0)
  {
    objectDetectionPublisher_.publish(detected_persons_tuw);
  }
  boundingBoxesResults_.bounding_boxes.clear();
  for (int i = 0; i < numClasses_; i++) {
    rosBoxes_[i].clear();
    rosBoxCounter_[i] = 0;
  }

  return 0;
}

void YoloObjectDetector::groundPlaneCallback(const rwth_perception_people_msgs::GroundPlane::ConstPtr &gp)
{
  gp_ = gp;

  // ground plane normal vector
  gpn_(0) = gp_->n[0];
  gpn_(1) = gp_->n[1];
  gpn_(2) = gp_->n[2];

  // ground plane distance
  gpd_ = ((float)gp_->d) * (-1.0);
}

} /* namespace darknet_ros*/

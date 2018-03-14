# ROS-deep-learning
Deep-learning nodes for ROS with support for NVIDIA Jetson TX1/TX2 and TensorRT

## Requirements
Requires https://github.com/dusty-nv/jetson-inference  
```
git clone http://github.com/dusty-nv/jetson-inference  
cd jetson-inference
mkdir build
cd build
cmake ../
make
sudo make install
```
For use on other platforms than jetson, adjust GPU architecture in CMakeLists.txt:
```
-gencode arch=compute_53,code=sm_53
```
For reference see: https://github.com/dusty-nv/jetson-inference/issues/35

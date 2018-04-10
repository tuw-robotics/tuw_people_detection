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

Also if some gstreamer or glib header is not found, change the include directories in the
jetson-inference CMakeLists.txt in line 58:
```
include_directories(/usr/lib/x86_64-linux-gnu/gstreamer-1.0/include/ /usr/include/gstreamer-1.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include /usr/lib/x86_64-linux-gnu/glib-2.0/include/ /usr/include/libxml2 /usr/lib/aarch64-linux-gnu/glib-2.0/include/ /usr/include/glib-2.0/)

```

For reference see: https://github.com/dusty-nv/jetson-inference/issues/35

## Config
The paths to the ```.prototxt```, respectively ```.caffemodel``` files have to be configured in `config/paths_detectnet.yaml`.
The files are located in the `jetson-inference` build directory depending on the architecture.

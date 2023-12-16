cmake_minimum_required(VERSION 2.6)

project(FairMOT)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/home/zmw/TensorRT-8.6.1.6/include/)
link_directories(/home/zmw/TensorRT-8.6.1.6/lib/)
# eigen
include_directories(/usr/local/include/eigen3/)
#ROS
include_directories(/opt/ros/noetic/include)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(FairMOT src/FairMOT.cpp src/detection.cpp src/Hungarian.cpp src/matching.cpp src/tracker.cpp src/utils.cpp src/kalmanfilter.cpp)

target_link_libraries(FairMOT nvinfer)
target_link_libraries(FairMOT cudart)
target_link_libraries(FairMOT nvinfer_plugin)
target_link_libraries(FairMOT nvonnxparser)
target_link_libraries(FairMOT ${OpenCV_LIBS})

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)



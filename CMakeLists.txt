PROJECT(cudalanedet LANGUAGES CXX CUDA)
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
ENABLE_LANGUAGE(CUDA)

MESSAGE(STATUS "Compiling CUDA compatible Hough Transform Lane Detector")
MESSAGE(STATUS "Using compiler: ${CMAKE_CXX_COMPILER}")
MESSAGE(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "Arch: ${CMAKE_SYSTEM_PROCESSOR}")

SET(CMAKE_CUDA_FLAGS "-std=c++14")
SET(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wno-unknown-pragmas -Werror")
SET(CMAKE CXX_STANDARD_REQUIRED True)

IF(${CMAKE_BUILD_TYPE} MATCHES "Release")
  MESSAGE(STATUS "Additional Flags for Release mode")
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp -O2 -DNDEBUG")
ELSEIF(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  MESSAGE(STATUS "Additional Flags for Debug mode")
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g")
ENDIF()

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(PCL 1.10 REQUIRED)
FIND_PACKAGE(PkgConfig)
FIND_PACKAGE(Armadillo REQUIRED)

pkg_check_modules(YAMLCPP REQUIRED yaml-cpp>=0.5)
include_directories(${YAMLCPP_INCLUDE_DIRS})
include_directories(${ARMADILLO_INCLUDE_DIRS})

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
message(STATUS "PCL Definitions: ${PCL_DEFINITIONS}")

INCLUDE_DIRECTORIES(include
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        # ${PCL_INCLUDE_DIRS}
)

# LINK_DIRECTORIES(include
#         ${PCL_INCLUDE_DIRS}
# )

INCLUDE($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
ROSBUILD_INIT()
SET(ROS_BUILD_STATIC_LIBS true)
SET(ROS_BUILD_SHARED_LIBS false)

MESSAGE(STATUS "ROS-Overrride Build Type: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
MESSAGE(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
# SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

SET(libs roslib roscpp rosbag glog gflags boost_system lua5.1
    pthread ${OpenCV_LIBS} yaml-cpp ${PCL_LIBRARIES} cv_bridge) 

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_75,code=sm_75 )
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_86,code=sm_86 )
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++14)

# CUDA_ADD_LIBRARY(houghtransformlib "HoughTransform.cu")
ADD_LIBRARY(houghtransformlib "HoughTransform.cu")

ROSBUILD_ADD_EXECUTABLE(cudalanedet
                        LaneDetection.cpp
                        Line.cpp
                        Preprocessing.cpp)

TARGET_LINK_LIBRARIES(cudalanedet houghtransformlib ${libs})
TARGET_LINK_LIBRARIES(cudalanedet ${YAMLCPP_LIBRARIES} ${ARMADILLO_LIBRARIES})

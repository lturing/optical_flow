cmake_minimum_required(VERSION 2.8)

project(optical_flow)




SET(CMAKE_BUILD_TYPE Debug )

# SET(CMAKE_BUILD_TYPE Release )

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}     -Wall -O3 -march=native -Wno-reorder")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native -Wno-reorder")

# Check C++17 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
add_definitions(-DCOMPILEDWITHC11)
message(STATUS "Using flag -std=c++17.")


option(PVIO_DEBUG "..." ON)
option(PVIO_ENABLE_FORENSICS "..." ON)
option(PVIO_ENABLE_THREADING "..." OFF)
option(PVIO_ENABLE_PLANE_CONSTRAINT "..." ON)



find_package(OpenCV 4.4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4.4 not found.")
   endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)


include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)



find_package(yaml-cpp REQUIRED)


add_library(${PROJECT_NAME} SHARED  
src/config.cc  
src/frame.cc 
src/opencv_image.cc  
src/track.cc  
src/yaml_config.cc  
include/common.h   
include/frame.h  
include/opencv_image.h   
include/pvio.h    
include/track.h    
include/yaml_config.h  
include/poisson_disk_filter.h
)


target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${YAML_CPP_LIBRARIES}
-lboost_serialization
-lcrypto
)




add_executable(opticalFlow  src/main.cc)
target_link_libraries(opticalFlow ${PROJECT_NAME})




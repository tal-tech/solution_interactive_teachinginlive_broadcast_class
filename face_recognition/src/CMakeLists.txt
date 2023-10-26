cmake_minimum_required(VERSION 3.10)

project(GodeyeVideoCrow)
set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/package")
#set(PACKVERSION 2.0.0.1)
######################################################################
#全局编译标志
######################################################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

set(CUDA_cublas_device_LIBRARY /usr/local/cuda/lib64)
######################################################################
#全局链接和包含
######################################################################

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/3rdParty)
include_directories(${CMAKE_SOURCE_DIR}/det_face_id/include)
link_directories(${CMAKE_SOURCE_DIR}/det_face_id/libs/linux/gpu)
link_directories(${CMAKE_SOURCE_DIR}/3rdParty/ubuntu-cuda10)

link_directories(${CMAKE_SOURCE_DIR}/3rdParty/opencv/lib)
set(OpenCV_INCLUDES ${PROJECT_SOURCE_DIR}/3rdParty/opencv/include)
file(GLOB OpenCV_LIBRARYS ${PROJECT_SOURCE_DIR}/3rdParty/opencv/lib/*.so*)


file(GLOB Boost_other /usr/lib/x86_64-linux-gnu/libboost*)
# # 找不到 chrono date_time atomic 库，指定位置从下面找
link_directories(/usr/lib/x86_64-linux-gnu/)


#################################################################L#####
#全局编译输出指定
######################################################################

#IF(CMAKE_BUILD_TYPE MATCHES Debug)
set(CMAKE_VERBOSE_MAKEFILE ON)
#ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

# string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/${BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/${BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${BUILD_TYPE})


######################################################################
#子工程目录
######################################################################
# add_subdirectory(base/third_party/dynamic_annotations)
# add_subdirectory(base/third_party/symbolize)
# add_subdirectory(base/third_party/xdg_mime)
# add_subdirectory(base/third_party/libevent)
# add_subdirectory(third_party)
# add_subdirectory(base)
# add_subdirectory(base/i18n)
# add_subdirectory(eureka)
# add_subdirectory(grpc)
# add_subdirectory(breakpad)
# add_subdirectory(det_face_mtcnn)
# add_subdirectory(caffe_inference_base)
# add_subdirectory(common/hiredis)

######################################################################
#编译对象
######################################################################

add_executable(GodeyeVideoCrow
    demo.cpp
    det_face_id.hpp
    det_face_mtcnn.hpp
)

target_include_directories(GodeyeVideoCrow PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdParty/opencv/include
)

target_link_libraries(GodeyeVideoCrow
    det_face_mtcnn
    det_face_id
    pthread
    ${Boost_other}
    ${OpenCV_LIBRARYS}
    ${Boost_other}
)

# add_dependencies(GodeyeVideoCrow facethink_caffe_GPU)
######################################################################
#程序发布设置
######################################################################

install(DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ DESTINATION bin)
install(DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/ DESTINATION lib)
install(DIRECTORY 3rdParty/tensorflow/r1.8/libs/Linux/cpu DESTINATION lib)
install(DIRECTORY 3rdParty/opencv/lib DESTINATION .)
install(DIRECTORY ai_model DESTINATION bin)
install(DIRECTORY det_face_id DESTINATION .)
install(DIRECTORY 3rdParty/ubuntu-cuda10 DESTINATION lib)

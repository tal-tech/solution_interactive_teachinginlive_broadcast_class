
cmake_minimum_required(VERSION 3.5)
project(det_face_mtcnn)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -fPIC -ggdb")

if (NOT WIN32)
  SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -fPIC")
   SET(CMAKE_CXX_FLAGS "-fPIC -Bsymbolic -fvisibility=hidden")
  SET(CMAKE_C_FLAGS "-fPIC -Bsymbolic -fvisibility=hidden")
endif()
option (CPU_ONLY "USE CPU_ONLY FOR CAFFE INFERENCE" OFF)

if (CPU_ONLY)
  set(PLATFORM_TYPE "CPU")
  add_definitions(-DCPU_ONLY)
else()
   set(PLATFORM_TYPE "GPU")
 endif()

#set(LIBRARY_NAME det_face_mtcnn_${PLATFORM_TYPE})
set(LIBRARY_NAME det_face_mtcnn)


if (WIN32)
  set(Boost_USE_STATIC_LIBS ON)  # static boost
endif()

include(cmake/CheckDependencies.cmake)
detect_all()

set(PROJECT_INCLUDE  ${PROJECT_INCLUDE} ${PROJECT_SOURCE_DIR}/api/include)
set(PROJECT_LIBRARY  ${PROJECT_LIBRARY})
include_directories(${PROJECT_INCLUDE})

#if (WIN32)
#  add_definitions(-DNO_STRICT)
#endif()

if (WIN32)
  list(APPEND CUDA_NVCC_FLAGS " -Xcompiler \"/wd 4819\"")
  list(APPEND CUDA_NVCC_FLAGS " -Xcompiler \"/MD\"")
  add_definitions(-DNO_STRICT)
  # add_definitions(-DBOOST_ALL_DYN_LINK)  #only for dynamic library
  add_definitions(-DDLL_EXPORTS)
else()
  list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -Xcompiler -Wno-deprecated")
  add_definitions(-DBOOST_ALL_DYN_LINK)
endif()

add_subdirectory(./api)
#add_subdirectory(./test)

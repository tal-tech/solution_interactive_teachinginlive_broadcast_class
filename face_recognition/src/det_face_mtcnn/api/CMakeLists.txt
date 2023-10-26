file(GLOB_RECURSE srcs *.cpp *.cc)

#add_library(${LIBRARY_NAME} STATIC ${srcs})
add_library(${LIBRARY_NAME} SHARED ${srcs})
target_link_libraries(${LIBRARY_NAME} ${PROJECT_LIBRARY} facethink_caffe_GPU)

if (NOT WIN32)
  find_package( Threads REQUIRED )
  target_link_libraries(${LIBRARY_NAME} ${PROJECT_LIBRARY} Threads::Threads)

  if(DEFINED VERSION)
    SET_TARGET_PROPERTIES(${LIBRARY_NAME} PROPERTIES VERSION ${VERSION})
  endif()
endif()

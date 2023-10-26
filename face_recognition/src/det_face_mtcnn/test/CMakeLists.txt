link_directories(${PROJECT_BINARY_DIR}/lib)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)



if (WIN32)
add_executable(demo_${PLATFORM_TYPE} demo.cpp  )
target_link_libraries(demo_${PLATFORM_TYPE} ${LIBRARY_NAME} ${PROJECT_LIBRARY}) 
    
add_executable(plugin_demo_${PLATFORM_TYPE} plugin_demo.cpp  )
target_link_libraries(plugin_demo_${PLATFORM_TYPE} ${LIBRARY_NAME} ${PROJECT_LIBRARY}) 
    
add_executable(performance_testing_${PLATFORM_TYPE} performance_testing.cpp)
target_link_libraries(performance_testing_${PLATFORM_TYPE} ${LIBRARY_NAME} ${PROJECT_LIBRARY})

else()
add_executable(demo_${PLATFORM_TYPE} demo.cpp  )
target_link_libraries(demo_${PLATFORM_TYPE} ${LIBRARY_NAME} ${PROJECT_LIBRARY} -ldl) 
    
add_executable(plugin_demo_${PLATFORM_TYPE} plugin_demo.cpp  )
target_link_libraries(plugin_demo_${PLATFORM_TYPE} ${LIBRARY_NAME} ${PROJECT_LIBRARY} -ldl) 
    
add_executable(performance_testing_${PLATFORM_TYPE} performance_testing.cpp)
target_link_libraries(performance_testing_${PLATFORM_TYPE} ${LIBRARY_NAME} ${PROJECT_LIBRARY} -ldl)
endif()
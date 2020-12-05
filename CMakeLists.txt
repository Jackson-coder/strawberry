cmake_minimum_required(VERSION 3.0.0)
project(camera VERSION 0.1.0)

find_package(OpenCV REQUIRED)
find_package(realsense2 2.39.0)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${realsense2_INCLUDE_DIR} )

add_executable(camera main.cpp)

target_link_libraries(camera ${OpenCV_LIBS} )
target_link_libraries(camera ${realsense2_LIBRARY} )
target_link_libraries(camera glfw )
target_link_libraries(camera OpenGL::GL)

cmake_minimum_required(VERSION 2.8)
project( vid2frame )
find_package( OpenCV REQUIRED )
add_executable( vid2frame vid2frame.cpp )
target_link_libraries( vid2frame ${OpenCV_LIBS} )

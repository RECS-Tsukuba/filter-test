cmake_minimum_required(VERSION 2.8)

project(linear_filter)
find_package(OpenCV REQUIRED)
add_executable(linear_filter linear_filter.cc)
target_link_libraries(linear_filter ${OpenCV_LIBS})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_CXX_COMPILER g++)
  set(CMAKE_CXX_FLAGS "-Wall")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif(CMAKE_SYSTEM_NAME STREQUAL "Linux")


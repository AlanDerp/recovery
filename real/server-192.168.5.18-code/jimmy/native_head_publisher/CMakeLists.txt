cmake_minimum_required(VERSION 3.16)
project(jimmy_native_head_publisher)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(jimmy_compressed_publisher compressed_publisher.cpp)
target_compile_features(jimmy_compressed_publisher PRIVATE cxx_std_17)
target_compile_options(jimmy_compressed_publisher PRIVATE -O3 -Wall -Wextra)
ament_target_dependencies(jimmy_compressed_publisher rclcpp sensor_msgs)

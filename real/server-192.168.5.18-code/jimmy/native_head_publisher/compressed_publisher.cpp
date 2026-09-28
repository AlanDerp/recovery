#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

namespace
{

bool read_exact(void *destination, std::size_t size)
{
  auto *output = static_cast<std::uint8_t *>(destination);
  std::size_t received = 0;
  while (received < size) {
    const ssize_t result = ::read(STDIN_FILENO, output + received, size - received);
    if (result == 0) {
      return false;
    }
    if (result < 0) {
      if (errno == EINTR) {
        continue;
      }
      std::cerr << "stdin read failed: " << std::strerror(errno) << '\n';
      return false;
    }
    received += static_cast<std::size_t>(result);
  }
  return true;
}

struct FrameHeader
{
  std::uint64_t timestamp_ns;
  std::uint32_t jpeg_size;
} __attribute__((packed));

}  // namespace

int main(int argc, char **argv)
{
  if (argc < 3) {
    std::cerr << "usage: jimmy_compressed_publisher TOPIC NODE_NAME\n";
    return 2;
  }
  const std::string topic = argv[1];
  const std::string node_name = argv[2];

  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>(node_name);
  auto qos = rclcpp::QoS(rclcpp::KeepLast(1))
    .best_effort()
    .durability_volatile();
  auto publisher =
    node->create_publisher<sensor_msgs::msg::CompressedImage>(topic, qos);

  RCLCPP_INFO(node->get_logger(), "publishing framed JPEG input to %s", topic.c_str());

  std::uint64_t count = 0;
  while (rclcpp::ok()) {
    FrameHeader header{};
    if (!read_exact(&header, sizeof(header))) {
      break;
    }
    if (header.jpeg_size == 0 || header.jpeg_size > 20U * 1024U * 1024U) {
      RCLCPP_ERROR(
        node->get_logger(), "invalid JPEG frame size: %u", header.jpeg_size);
      break;
    }

    sensor_msgs::msg::CompressedImage message;
    message.header.stamp.sec =
      static_cast<std::int32_t>(header.timestamp_ns / 1000000000ULL);
    message.header.stamp.nanosec =
      static_cast<std::uint32_t>(header.timestamp_ns % 1000000000ULL);
    message.format = "jpeg";
    message.data.resize(header.jpeg_size);
    if (!read_exact(message.data.data(), message.data.size())) {
      break;
    }

    publisher->publish(std::move(message));
    ++count;
    if ((count % 300U) == 0U) {
      RCLCPP_INFO(node->get_logger(), "published %lu frames", count);
    }
    rclcpp::spin_some(node);
  }

  RCLCPP_INFO(node->get_logger(), "stopped after %lu frames", count);
  rclcpp::shutdown();
  return 0;
}

#!/bin/bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/nvidia/galaxea/install/setup.bash

cmake \
  -S /home/nvidia/jimmy/native_head_publisher \
  -B /home/nvidia/jimmy/native_head_publisher/build \
  -DCMAKE_BUILD_TYPE=Release

cmake --build /home/nvidia/jimmy/native_head_publisher/build --parallel 2

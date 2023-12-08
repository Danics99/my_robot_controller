#!/bin/bash

# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Source your ROS2 workspace
source /home/dani/ros2_ws/install/setup.bash

# Start MicroXRCEAgent
gnome-terminal -- bash -c "MicroXRCEAgent udp4 -p 8888; exec bash"

# Start PX4 SITL with a delay to ensure it's fully up and running
gnome-terminal --working-directory=/home/dani/PX4-Autopilot -- bash -c "make px4_sitl gz_x500; sleep 10; exec bash"

# Start ROS2 nodes with a delay to ensure PX4 SITL is ready
gnome-terminal -- bash -c "sleep 5; source /opt/ros/humble/setup.bash; source /home/dani/ros2_ws/install/setup.bash; ros2 run my_robot_controller audio_recognizer_node; exec bash"
gnome-terminal -- bash -c "sleep 10; source /opt/ros/humble/setup.bash; source /home/dani/ros2_ws/install/setup.bash; ros2 run my_robot_controller uav_control_node; exec bash"

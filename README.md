# my_robot_controller
ROS2 python package to control the drone in Gazebo with voice commands for the Special Course project.

## Main resources:
- [ROS2](https://docs.ros.org/en/humble/index.html)
- [PX4](https://docs.px4.io/main/en/ros/ros2_comm.html)
- [whisper-mic](https://github.com/mallorbc/whisper_mic)

## Setup
1. Run ```trainBert.py``` to create BERT model
2. Run ```MicroXRCEAgent udp4 -p 8888``` to start MicroXRCEAgent
3. Run ```make px4_sitl gz_x500``` inside PX4-Autopilot folder to start PX4 SITL
4. Run ```ros2 run my_robot_controller audio_recognizer_node``` to start node that handles voice inputs
5. Run ```ros2 run my_robot_controller uav_control_node``` to start node that controls the drone with the commands given by the other node

## List of instructions
You can use the following instructions and synonyms with or without a given value for the action to control the drone. Example: "Go up 5 meters".
- Take off
- Land
- Go up
- Go down
- Go forward
- Go backward
- Move right
- Move left
- Turn right
- Turn left
- Stop


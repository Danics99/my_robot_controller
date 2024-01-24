# my_robot_controller
ROS2 python package to control the drone in Gazebo with voice commands for the Special Course project.

## Main resources:
- [ROS2](https://docs.ros.org/en/humble/index.html)
- [PX4](https://docs.px4.io/main/en/ros/ros2_comm.html)
- [whisper-mic](https://github.com/mallorbc/whisper_mic)

## Setup
Create a ROS2 workspace:
```
mkdir -p ~/ros2_ws/src/
cd ~/ros2_ws/src/
```

Clone px4_msgs and this repository to `\src` 
```
git clone https://github.com/PX4/px4_msgs.git
git clone https://github.com/Danics99/my_robot_controller.git
```

Source the ROS 2 development environment and the local_setup.bash into the current terminal:
```
cd ..
source /opt/ros/humble/setup.bash
source install/local_setup.bash
colcon build
```

Compile the workspace using colcon:
```
colcon build
```

## How to use
1. Run trainBert.py to create BERT model
2. In one terminal, run ```MicroXRCEAgent udp4 -p 8888``` to start MicroXRCEAgent
3. To start PX4 SITL, open another terminal and run:
   ```
   cd ~/cd PX4-Autopilot/
   make px4_sitl gz_x500
   ```  
4. To start the node that handles voice inputs, open another terminal and run:
   ```
   cd ~/ros2_ws/src/my_robot_controller/my_robot_controller/
   ros2 run my_robot_controller audio_recognizer_node
    ``` 
5. To start node that controls the drone with the commands given by the other node, open another terminal and run:
    ```
    cd ~/ros2_ws/src/my_robot_controller/my_robot_controller/
    ros2 run my_robot_controller uav_control_node
    ``` 

## List of instructions
| Take off       | Switch to offboard mode and launch the drone           |
| Land           | Switch to landing mode                                 |
| Go up          | Negative movement in the z-axis (NED frame)            |
| Go down        | Positive movement in the z-axis (NED frame)            |
| Go forward     | Movement in the direction the drone is facing          |
| Go backward    | Movement in the opposite direction the drone is facing |
| Move right     | Right lateral movement                                 |
| Move left      | Left lateral movement                                  |
| Turn right     | Right rotation around the z-axis                       |
| Turn left      | Left rotation around the z-axis                        |
| Stop           | Maintain current position                              |

You can use the instructions above and synonyms with or without a given value for the action to control the drone. Example: "Go up 5 meters".

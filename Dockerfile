# Use Ros humble as the base image
FROM ros:humble-ros-base-jammy
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc"

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-dev-tools \
    x11-apps \
    gnome-terminal \
    dbus-x11 \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    python3-pip \
    portaudio19-dev \
    python3-dev \
    build-essential \
    git \
    wget \
    lsb-release \
    software-properties-common \
    sudo \
    cmake \
    usbutils \
    mesa-utils \
    gpg \
    pulseaudio \
    && rm -rf /var/lib/apt/lists/*

# Set the DISPLAY environment variable
ENV DISPLAY=host.docker.internal:0.0

# Create workspace directory and set it as working directory
WORKDIR /home

# PX4 Autopilot
RUN git clone https://github.com/PX4/PX4-Autopilot.git --recursive
RUN apt-get update && apt-get install -y sudo
RUN ./PX4-Autopilot/Tools/setup/ubuntu.sh && make -C PX4-Autopilot px4_sitl

# Micro-XRCE-DDS-Agent
RUN git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git && \
    cd Micro-XRCE-DDS-Agent && \
    mkdir build && cd build && cmake .. && make && \
    sudo make install && sudo ldconfig /usr/local/lib/

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create base ROS workspace
RUN mkdir -p /home/ros2_ws/src/ && cd /home/ros2_ws/src && \
    git clone https://github.com/PX4/px4_msgs.git && \
    git clone https://github.com/Danics99/my_robot_controller.git && \
    . /opt/ros/humble/setup.sh && \ 
    cd .. && colcon build

# Copy data into the container
COPY /my_robot_controller/bert_model /home/ros2_ws/src/my_robot_controller/my_robot_controller/
COPY /my_robot_controller/~ /home/ros2_ws/src/my_robot_controller/my_robot_controller/
COPY /my_robot_controller/start_drone_simulation.sh /home/ros2_ws/src/my_robot_controller/my_robot_controller/
RUN cd /home/ros2_ws/src/my_robot_controller/my_robot_controller/ && chmod +x start_drone_simulation.sh
    
# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the default command
CMD ["bash"]

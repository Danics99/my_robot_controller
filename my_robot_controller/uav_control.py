import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
import math

class DroneControllerNode(Node):
    def __init__(self):
        super().__init__('drone_controller_node')

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.is_takeoff_commanded = False
        self.is_landing_commanded = False  

        # Initialize target setpoint variables with None
        self.target_x = None
        self.target_y = None
        self.target_z = None
        self.target_yaw = None

        # Save initial altitude
        self.initial_z = self.vehicle_local_position.z  

        # Add flags for operations
        self.is_setpoint_published = False
        self.is_setpoint_reached = False
        self.new_setpoint_issued = True 

        # Subscribers
        self.subscription = self.create_subscription(
            String,
            'recognized_instruction',
            self.listener_callback,
            10)
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        
        # Publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        
        # Create a timer to publish control commands
        self.timer = self.create_timer(0.1, self.timer_callback)

    def vehicle_local_position_callback(self, vehicle_local_position):
        self.vehicle_local_position = vehicle_local_position

        # Initialize target setpoints with current position if they are None
        if self.target_x is None:
            self.target_x = self.vehicle_local_position.x
        if self.target_y is None:
            self.target_y = self.vehicle_local_position.y
        if self.target_z is None:
            self.target_z = self.vehicle_local_position.z
        if self.target_yaw is None:
            self.target_yaw = self.vehicle_local_position.heading

        # Check if the target setpoint is reached within a tolerance
        tolerance = 0.1 
        if self.is_setpoint_reached and not self.new_setpoint_issued:
            return  # Avoid logging if setpoint is already reached and no new setpoint is issued

        if (abs(self.vehicle_local_position.x - self.target_x) < tolerance and
            abs(self.vehicle_local_position.y - self.target_y) < tolerance and
            abs(self.vehicle_local_position.z - self.target_z) < tolerance and
            abs(self.vehicle_local_position.heading - self.target_yaw) < tolerance):
            if not self.is_setpoint_reached:
                self.is_setpoint_reached = True
                self.new_setpoint_issued = False
                self.get_logger().info("Target setpoint reached")
        else:
            self.is_setpoint_reached = False

    def listener_callback(self, msg):
    # Process the instruction
        instruction, value = self.process_instruction(msg.data)

        # Execute the command
        if hasattr(self, instruction):
            command_method = getattr(self, instruction)
            # Check if the command requires a value
            if instruction in ['land', 'stop', 'engage_offboard_mode', 'arm', 'disarm']:
                command_method() 
            else:
                command_method(value)
        else:
            self.get_logger().error(f"Unrecognized instruction: {instruction}")

    def process_instruction(self, instruction_str):
        # Define a mapping of multi-word instructions
        multi_word_instructions = {
            "take off": "take_off", 
            "go up": "go_up",
            "go down": "go_down",
            "go forward": "go_forward",
            "go backward": "go_backward",
            "move right": "move_right",
            "move left": "move_left",
            "turn right": "turn_right",
            "turn left": "turn_left",
        }

        # Check if the instruction is a known multi-word instruction
        for multi_word, single_word in multi_word_instructions.items():
            if instruction_str.startswith(multi_word):
                # Replace the multi-word instruction with a single-word equivalent
                instruction_str = instruction_str.replace(multi_word, single_word, 1)
                break

        parts = instruction_str.split()
        instruction = parts[0]
        value = None

        # Get the default value for the instruction, if it exists
        instructions_with_defaults = {
            "take_off": 1.0,    # Default height for takeoff
            "go_up": 1.0,       # Default distance to go up
            "go_down": 1.0,     # Default distance to go down
            "go_forward": 1.0,  # Default distance to go forward
            "go_backward": 1.0, # Default distance to go backward
            "move_right": 1.0,  # Default distance to move right
            "move_left": 1.0,   # Default distance to move left
            "turn_right": 30.0, # Default angle to turn right
            "turn_left": 30.0,  # Default angle to turn left
        }

        # Get the default value for the instruction, if it exists
        value = instructions_with_defaults.get(instruction)

        # If a specific value is provided in the message, override the default
        if len(parts) > 1:
            try:
                value = float(parts[1])
            except ValueError:
                self.get_logger().error(f"Invalid value for {instruction}: {parts[1]}")
                return instruction, None

        return instruction, value

    def vehicle_status_callback(self, vehicle_status):
        # Callback function for vehicle_status topic subscriber
        self.vehicle_status = vehicle_status

    def engage_offboard_mode(self):
        # Switch to offboard mode.
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def arm(self):
        # Send an arm command to the vehicle
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info("Arming drone.")

    def disarm(self):
        # Send a disarm command to the vehicle
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def publish_offboard_control_heartbeat_signal(self):
        # Publish the offboard control mode
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_setpoint(self):
        # Check if any of the target setpoints are None
        if None in [self.target_x, self.target_y, self.target_z, self.target_yaw]:
            self.get_logger().warn("Setpoint targets not initialized. Skipping setpoint publication.")
            return
        
        if not self.is_setpoint_published:
            self.is_setpoint_published = True
            self.get_logger().info(f"Publishing setpoint at position {[self.target_x, self.target_y, self.target_z]} with yaw {self.target_yaw}")

        # Publish the target setpoint
        msg = TrajectorySetpoint()
        msg.position = [self.target_x, self.target_y, self.target_z]
        msg.yaw = self.normalize_angle(self.target_yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, **params) -> None:
        # Publish a vehicle command 
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def take_off(self, value=1.0):
        if self.vehicle_status.pre_flight_checks_pass:
            # Check if the drone is armed
            if self.vehicle_status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.get_logger().info("Drone is not armed.")
                self.arm()
            else:
                pass
        else: 
            self.get_logger().info("Drone can't be armed.")

        self.get_logger().info(f"Drone armed")

        # Set the target setpoint for takeoff
        self.target_z -= value
        self.update_setpoint_flags() 
        self.get_logger().info(f"Takeoff to {value} meters.")

    def land(self):
        # Send a command to land
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Landing drone.")
        self.disarm()
        exit(0)

    def go_up(self, value=1.0):
        self.target_z -= value
        self.update_setpoint_flags()  
        self.get_logger().info(f"Going up {value} meters.")

    def go_down(self, value=1.0):
        self.target_z += value
        self.update_setpoint_flags()
        self.get_logger().info(f"Going down {value} meters.")

    def go_forward(self, value=1.0):
        self.target_x += value * math.cos(self.normalize_angle(self.vehicle_local_position.heading))
        self.target_y += value * math.sin(self.normalize_angle(self.vehicle_local_position.heading))
        self.update_setpoint_flags()
        self.get_logger().info(f"Going forward {value} meters.")

    def go_backward(self, value=1.0):
        self.target_x -= value * math.cos(self.normalize_angle(self.vehicle_local_position.heading))
        self.target_y -= value * math.sin(self.normalize_angle(self.vehicle_local_position.heading))
        self.update_setpoint_flags()
        self.get_logger().info(f"Going backward {value} meters.")

    def move_right(self, value=1.0):
        self.target_x -= value * math.sin(self.normalize_angle(self.vehicle_local_position.heading))
        self.target_y += value * math.cos(self.normalize_angle(self.vehicle_local_position.heading))
        self.update_setpoint_flags()
        self.get_logger().info(f"Moving right {value} meters.")

    def move_left(self, value=1.0):
        self.target_x += value * math.sin(self.normalize_angle(self.vehicle_local_position.heading))
        self.target_y -= value * math.cos(self.normalize_angle(self.vehicle_local_position.heading))
        self.update_setpoint_flags()
        self.get_logger().info(f"Moving left {value} meters.")

    def normalize_angle(self, angle):
        # Normalize an angle to the range [-pi, pi)
        normalized_angle = (angle + math.pi) % (2 * math.pi) - math.pi
        return normalized_angle

    def turn_right(self, value=30.0):
        # Convert degrees to radians and add to current yaw
        self.target_yaw += math.radians(value)
        # Normalize the angle
        self.target_yaw = self.normalize_angle(self.target_yaw)
        self.update_setpoint_flags() 
        self.get_logger().info(f"Turning right {value} degrees.")

    def turn_left(self, value=30.0):
        # Convert degrees to radians and subtract from current yaw
        self.target_yaw -= math.radians(value)
        # Normalize the angle
        self.target_yaw = self.normalize_angle(self.target_yaw)
        self.update_setpoint_flags()
        self.get_logger().info(f"Turning left {value} degrees.")

    def update_setpoint_flags(self):
        self.is_setpoint_published = False
        self.is_setpoint_reached = False
        self.new_setpoint_issued = True

    def stop(self):
        # Set the target setpoints to the current position
        self.target_x = self.vehicle_local_position.x
        self.target_y = self.vehicle_local_position.y
        self.target_z = self.vehicle_local_position.z
        self.target_yaw = self.vehicle_local_position.heading
        self.update_setpoint_flags()
        self.get_logger().info(f"Stopping and maintaining position at {[self.target_x, self.target_y, self.target_z]} with yaw {self.target_yaw}")

    def timer_callback(self) -> None:
        self.publish_offboard_control_heartbeat_signal()
        self.publish_setpoint()

        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1
        
def main(args=None):
    rclpy.init(args=args)
    node = DroneControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)

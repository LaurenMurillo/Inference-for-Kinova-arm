#!/usr/bin/env python3
"""
OpenPI Kinova Direct Bridge Node

This node directly loads and runs the OpenPI PyTorch model within ROS2,
eliminating the need for a separate policy server. It sends commands directly
to the robot via the Kortex API.

Enhanced with rosbag-compatible data processing to ensure training/inference consistency.
"""

# Print the python binary location and paths
import sys
import traceback
print(f"Python binary: {sys.executable}")
print(f"Python sys.path: {sys.path}")

#rclpy - (ROS Client Library for Python) is the official Python API for ROS 2, 
#enabling developers to interact with the Robot Operating System 2 environment
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import numpy as np
import cv2
from cv_bridge import CvBridge
import threading
import time
import os
import sys
from typing import Dict, Optional, Any, List, Tuple
import pathlib
import torch
import pickle
import base64

# ROS2 message types
from sensor_msgs.msg import Image, JointState
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import String, Bool, Header, Float32MultiArray, Float32
from control_msgs.msg import DynamicJointState, JointTrajectoryControllerState, GripperCommand
from tf2_msgs.msg import TFMessage

# Kinova Kortex API imports
try:
    # Check if we should use mock API for simulation
    # We need to detect this before node initialization since imports happen at module level
    simulation_mode = False  # Default
    if '--ros-args' in sys.argv:
        # Try to extract simulation_mode from args (before node init)
        # Look for -p simulation_mode:=true
        for i, arg in enumerate(sys.argv):
            if 'simulation_mode:=true' in arg:
                simulation_mode = True
                break
            elif arg == '-p' and i+1 < len(sys.argv) and 'simulation_mode:=true' in sys.argv[i+1]:
                simulation_mode = True
                break
    
    if simulation_mode:
        # Use mock API for simulation
        print("="*70)
        print(" SIMULATION MODE: Using Mock Kortex API")
        print(" Commands will be published to ros2_control")
        print("="*70)
        from mock_kortex_api import (
            BaseClient, BaseCyclicClient, TCPTransport, UDPTransport,
            RouterClient, SessionManager, ActuatorConfigClient,
            BaseCyclic_pb2, Session_pb2, Base_pb2, ActuatorConfig_pb2,
            RouterClientSendOptions
        )
    else:
        # Use real API for hardware
        print("="*70)
        print(" HARDWARE MODE: Using Real Kortex API")
        print(" Commands will be sent via TCP/UDP to robot")
        print("="*70)
        from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
        from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
        from kortex_api.autogen.messages import BaseCyclic_pb2, Session_pb2, Base_pb2, ActuatorConfig_pb2
        from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
        from kortex_api.SessionManager import SessionManager
        from kortex_api.UDPTransport import UDPTransport
        from kortex_api.TCPTransport import TCPTransport
        from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
    
    KORTEX_API_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Kortex API not available: {e}")
    print("Make sure Kortex API is installed (or mock_kortex_api.py for simulation)")
    KORTEX_API_AVAILABLE = False

# OpenPI imports - add the openpi path
try:
    # Add OpenPI to Python path
    openpi_path = pathlib.Path.home() / "openpi" / "src"
    if openpi_path.exists():
        sys.path.insert(0, str(openpi_path))
    
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    from openpi.shared import download
    import openpi.transforms as transforms
    
    OPENPI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: OpenPI not available: {e}")
    print("Make sure OpenPI is installed and accessible")
    OPENPI_AVAILABLE = False

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 1

# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check


# --- Data Processing Logic (from rosbag_to_cobot_dataset.py) ---

class DataProcessor:
    """Implements the same data transformation logic as rosbag_to_cobot_dataset.py"""

    def __init__(self):
        self.bridge = CvBridge()
        self.joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4", 
            "joint_5", "joint_6", "joint_7"
        ]
        self.gripper_joint_name = "gripper_joint"
        self.target_image_size = (256, 256)

    def _process_image(self, img_msg: Image) -> np.ndarray:
        """Convert ROS Image message to numpy array, resize, and convert to RGB."""
        try:
            # Prefer 'bgr8' to handle most common camera outputs, then convert to RGB
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Resize to standard size (256, 256) as in the extraction script
            cv_image = cv2.resize(cv_image, self.target_image_size)
            return cv_image.astype(np.uint8)
        except Exception as e:
            # Fallback to a zero array if image processing fails
            print(f"Error processing image: {e}")
            return np.zeros((*self.target_image_size, 3), dtype=np.uint8)

    def _extract_joint_state(self, joint_msg: JointState) -> np.ndarray:
        """Extract joint positions for the 7-DOF arm + gripper (8 values)."""
        joint_positions = np.zeros(8)  # 7 arm joints + 1 gripper
        
        for i, joint_name in enumerate(self.joint_names):
            try:
                idx = joint_msg.name.index(joint_name)
                joint_positions[i] = joint_msg.position[idx]
            except (ValueError, IndexError):
                joint_positions[i] = 0.0
        
        # Add gripper state
        try:
            gripper_idx = joint_msg.name.index(self.gripper_joint_name)
            joint_positions[7] = joint_msg.position[gripper_idx]
        except (ValueError, IndexError):
            joint_positions[7] = 0.0
        
        return joint_positions.astype(np.float32)

    def _extract_joint_velocities(self, dynamic_msg: Optional[DynamicJointState]) -> np.ndarray:
        """Extract joint velocities for 7-DOF arm (7 values)."""
        velocities = np.zeros(8)
        
        if dynamic_msg is None:
            return velocities.astype(np.float32)
            
        for i, joint_name in enumerate(self.joint_names):
            try:
                idx = dynamic_msg.joint_names.index(joint_name)
                if idx < len(dynamic_msg.interface_values):
                    # Look for velocity interface
                    interface_idx = None
                    for j, interface in enumerate(dynamic_msg.interface_values[idx].interface_names):
                        if 'velocity' in interface.lower():
                            interface_idx = j
                            break
                    if interface_idx is not None and interface_idx < len(dynamic_msg.interface_values[idx].values):
                        velocities[i] = dynamic_msg.interface_values[idx].values[interface_idx]
            except (ValueError, IndexError):
                velocities[i] = 0.0
        return velocities.astype(np.float32)

    def _extract_pose(self, controller_msg: Optional[JointTrajectoryControllerState]) -> np.ndarray:
        """Extract end-effector pose [x, y, z, qx, qy, qz, qw] (7 values)."""
        # NOTE: This is a simplified implementation. In practice, you might need
        # to get the actual end-effector pose from TF or forward kinematics
        pose = np.zeros(8)
        
        if controller_msg is not None and hasattr(controller_msg, 'feedback') and controller_msg.feedback:
            desired = controller_msg.feedback
            # Assuming the first 7 positions are pose-related data
            if hasattr(desired, 'positions') and len(desired.positions) >= 7:
                pose[:] = desired.positions[:7]

        return pose.astype(np.float32)

    def transform_for_inference(self, data: Dict[str, Any], prompt: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Processes and combines raw messages into the model's input tensors.
        
        Returns:
            Tuple of (vision_tensor, state_tensor) for the model
        """
        try:
            # 1. Image Processing
            rgb_img = self._process_image(data['rgb_image'])
            wrist_img = self._process_image(data['wrist_image'])
            
            # Convert to float, normalize, and move to C, H, W format
            img_tensor = torch.from_numpy(rgb_img).float() / 255.0  # HWC -> [256, 256, 3]
            wrist_tensor = torch.from_numpy(wrist_img).float() / 255.0
            
            # 2. State Processing
            joint_state = self._extract_joint_state(data['joint_states'])
            
            model_input = {
                    'image': rgb_img,
                    'wrist_image' : wrist_img,
                    'state': joint_state,
                    'prompt': prompt
                }

            return model_input
            
        except Exception as e:
            print(f"Error in transform_for_inference: {e}")
            return None


class OpenPIKinovaDirectBridge(Node):
    """ROS2 node that runs OpenPI model directly with Kortex API integration."""
    
    def __init__(self):
        super().__init__('openpi_kinova_direct_bridge')
        
        # Declare parameters
        self.declare_parameter('openpi_checkpoint_dir', 
    '/home/ros/pytorch_models/openpi/cobot_finetune_grab_can_150_eep_with_gripper__004/29999') #change checkpoint path (changed)
        self.declare_parameter('robot_ip', '192.168.1.10')  # Kinova robot IP
        self.declare_parameter('robot_port', 10000)  # Kinova robot port
        self.declare_parameter('robot_name', 'my_gen3')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('control_frequency', 10.0)  # 10Hz for effort control
        self.declare_parameter('image_width', 256)
        self.declare_parameter('image_height', 256)
        self.declare_parameter('default_prompt', 'Grab the can') #changed to grab the can
        self.declare_parameter('action_mode', 'pose')  # 'effort' for direct torque control, 'joint' for joint control, 'pose' for Cartesian control (changed to pose bc that's what our policy outputs)
        self.declare_parameter('effort_limit', 20.0)  # Maximum effort in Nm
        self.declare_parameter('estop_topic', '/arm_estop')  # E-stop status topic
        self.declare_parameter('estop_timeout', 0.045)  # Max time without e-stop signal (45ms allows ~28Hz with epsilon)
        self.declare_parameter('simulation_mode', False)  # Enable mock Kortex API for simulation
        
        # Get parameters
        self.robot_ip = self.get_parameter('robot_ip').value
        self.robot_port = self.get_parameter('robot_port').value
        self.robot_name = self.get_parameter('robot_name').value
        self.base_frame = self.get_parameter('base_frame').value
        self.control_freq = self.get_parameter('control_frequency').value
        self.img_width = self.get_parameter('image_width').value
        self.img_height = self.get_parameter('image_height').value
        self.default_prompt = self.get_parameter('default_prompt').value
        self.action_mode = self.get_parameter('action_mode').value
        self.effort_limit = self.get_parameter('effort_limit').value
        self.estop_topic = self.get_parameter('estop_topic').value
        self.estop_timeout = self.get_parameter('estop_timeout').value
        self.simulation_mode = self.get_parameter('simulation_mode').value
        
 
        # Initialize CV bridge and data processor
        self.cv_bridge = CvBridge()
        self.data_processor = DataProcessor()
        
        # State variables - enhanced for rosbag compatibility
        self.latest_data: Dict[str, Optional[Any]] = {
            'rgb_image': None,
            'wrist_image': None,
            'joint_states': None,
            'dynamic_joint_states': None,
            'controller_state': None,
            'tf_data': None,
        }
        
        # Legacy state variables (for backward compatibility)
        self.latest_joint_state = None
        self.latest_exterior_image = None
        self.latest_wrist_image = None
        self.latest_gripper_position = None
        self.current_prompt = self.default_prompt
        self.current_robot_state = None
        self.model_loaded = False
        
        # Kortex API variables
        self.kortex_connected = False
        self.router = None
        self.base_client = None
        self.base_cyclic_client = None
        self.session_manager = None
        self.cyclic_command = None
        self.current_efforts = np.zeros(7)  # 7 joints for Gen3
        
        # E-stop state tracking
        self.estop_active = False  # Start in e-stop mode for safety (changed to off), Add it back to True once proper e-stop hardware is wired up!
        self.last_estop_time = time.time()
        self.estop_watchdog_active = True
        self.estop_lock = threading.Lock()

        # Thread locks
        self.state_lock = threading.Lock()
        self.effort_lock = threading.Lock()
        
        # Model output variables
        self.fresh_actions = False
        self.current_actions = None
        self.fresh_actions_lock = threading.Lock()
        
        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers - Enhanced for rosbag compatibility
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            sensor_qos
        )
        
        # RGB camera (exterior view)
        self.rgb_image_sub = self.create_subscription(
            Image,
            '/rgb/image_raw',  # Primary topic from rosbag extraction (check if this is correct)
            self.rgb_image_callback,
            sensor_qos
        )
        
        # Fallback to alternative camera topic
        self.exterior_image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.exterior_image_callback,
            sensor_qos
        )
        
        # Wrist camera
        self.wrist_image_sub = self.create_subscription(
            Image,
            '/wrist_camera/color/image_raw',  # Alternative wrist camera topic (check if this is correct)
            self.wrist_image_callback,
            sensor_qos
        )
        
        # Alternative wrist camera topic from rosbag
        self.wrist_image_sub2 = self.create_subscription(
            Image,
            '/camera/color/image_raw',  # May be the same as exterior in some setups
            self.wrist_image_callback,
            sensor_qos
        )
        
        # Dynamic joint states (for velocities)
        self.dynamic_joint_sub = self.create_subscription(
            DynamicJointState,
            '/dynamic_joint_states',
            self.dynamic_joint_callback,
            sensor_qos
        )
        
        # Controller state (for pose information)
        self.controller_state_sub = self.create_subscription(
            JointTrajectoryControllerState,
            '/joint_trajectory_controller/state',
            self.controller_state_callback,
            sensor_qos
        )
        
        # TF data (for completeness, though not strictly required)
        self.tf_sub = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            sensor_qos
        )
        
        # Prompt subscription
        self.prompt_sub = self.create_subscription(
            String,
            '/openpi/prompt',
            self.prompt_callback,
            10
        )

        # Model Status Sub
        self.model_status_sub = self.create_subscription(
            Bool,
            '/openpi/model_status',
            self.model_status_callback,
            1
        )

        model_output_group = MutuallyExclusiveCallbackGroup()
        self.model_output_sub = self.create_subscription(
            String,
            '/openpi/model_output',
            self.model_output_callback,
            10,
            callback_group = model_output_group
        )
        
        # Arm command queue
        arm_eep_cb_group = MutuallyExclusiveCallbackGroup()
        self.arm_eep_sub = self.create_subscription(
            Float32MultiArray,
            '/openpi/arm_eep',
            self.arm_eep_callback,
            10,
            callback_group=arm_eep_cb_group)
        
        # Gripper command queue
        gripper_pos_cb_group = MutuallyExclusiveCallbackGroup()
        self.gripper_pos_sub = self.create_subscription(
            Float32,
            '/openpi/gripper_pos',
            self.gripper_pos_callback,
            10,
            callback_group=gripper_pos_cb_group)


        # E-stop status subscription (expects "true"/"false" strings at ~30Hz)
        # This replaces the old emergency stop subscription with a more robust watchdog system
        estop_cb_group = MutuallyExclusiveCallbackGroup()
        self.estop_status_sub = self.create_subscription(
            Bool,
            self.estop_topic,
            self.estop_status_callback,
            10,
            callback_group=estop_cb_group
        )
        
        # Publishers
        self.gripper_pub = self.create_publisher(
            GripperCommand,
            '/robotiq_gripper/gripper_cmd',
            10
        )
        
        # Status publishers
        self.status_pub = self.create_publisher(String, '/openpi/status', 10)
        self.plan_status_pub = self.create_publisher(String, '/openpi/plan_status', 10)
        self.effort_pub = self.create_publisher(Float32MultiArray, '/openpi/current_efforts', 10)
        
        self.model_input_pub = self.create_publisher(String, '/openpi/model_input', 10)

        self.arm_eep_pub = self.create_publisher(Float32MultiArray, '/openpi/arm_eep',10)
        self.gripper_pos_pub = self.create_publisher(Float32, '/openpi/gripper_pos', 10)
        
        # Initialize mock API if in simulation mode
        if self.simulation_mode and KORTEX_API_AVAILABLE:
            self.get_logger().info("Initializing Mock Kortex API with ROS node...")
            BaseCyclicClient.set_node(self)
        
        # Connect to Kinova robot via Kortex API
        self.connect_to_robot()
        
        # Control timer (start after model is loaded and robot connected)
        if self.kortex_connected:
            self.control_timer = self.create_timer(
                1.0 / self.control_freq,
                self.control_loop
            )
        
        # Start e-stop watchdog thread
        self.start_estop_watchdog()
        
        self.get_logger().info(f"OpenPI Kinova Direct Bridge initialized")
        self.get_logger().info(f"Robot IP: {self.robot_ip}:{self.robot_port}")
        self.get_logger().info(f"E-stop watchdog active, timeout: {self.estop_timeout}s")
     
    def connect_to_robot(self):
        """Connect to Kinova robot via Kortex API."""
        if not KORTEX_API_AVAILABLE:
            self.get_logger().error("Kortex API not available - cannot connect to robot")
            self.publish_status("kortex_unavailable")
            return
        
        # In simulation mode, skip actual robot connection checks
        if self.simulation_mode:
            self.get_logger().info("Simulation mode: Using mock Kortex API")
            self.get_logger().info("Mock connection will be established (no real robot required)")
        
        try:
            self.get_logger().info(f"Connecting to Kinova robot at {self.robot_ip}:{self.robot_port}")
            
            # Create connection objects
            self.transport = TCPTransport()
            self.transport_udp = UDPTransport()
            self.transport.connect(self.robot_ip, self.robot_port)
            self.transport_udp.connect(self.robot_ip, 10001)

            self.get_logger().info("Created Transports")
            
            self.router = RouterClient(self.transport, lambda: None)
            self.router_udp = RouterClient(self.transport_udp, lambda: None)
            self.get_logger().info("Created Routers")
            
            # Create session manager
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = "admin"
            session_info.password = "admin"
            session_info.session_inactivity_timeout = 10000   # (milliseconds)
            session_info.connection_inactivity_timeout = 2000 # (milliseconds)

            self.session_manager = SessionManager(self.router)
            self.actuator_config = ActuatorConfigClient(self.router)
            self.session_manager_udp = SessionManager(self.router_udp)
           
            self.session_manager.CreateSession(session_info)
            self.session_manager_udp.CreateSession(session_info)

            self.get_logger().info("Created Session Managers")
            
            # Create service clients
            self.base_client = BaseClient(self.router)
            self.base_cyclic_client = BaseCyclicClient(self.router_udp)

            self.get_logger().info("Created Base Clients")
            
            # Initialize cyclic command
            self.cyclic_command = BaseCyclic_pb2.Command()
            self.get_logger().info("Initialized Cyclic Command")
            
            # Set robot to low-level servoing mode for effort control
            if self.action_mode == 'effort':
                servoing_mode = Base_pb2.ServoingModeInformation()
                #servoing_mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
                servoing_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
                self.base_client.SetServoingMode(servoing_mode)
                self.get_logger().info("Set robot to low-level servoing mode for effort control")
            else:
                servoing_mode = Base_pb2.ServoingModeInformation()
                servoing_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
                self.base_client.SetServoingMode(servoing_mode)
                self.get_logger().info("Set robot to single-level servoing mode")

            # TEMP: HIGH LEVEL JOINT CONTROL
            """
            print("Starting angular action movement ...")
            actuator_count = self.base_client.GetActuatorCount()
            with open('/home/ros/cobot_ws/temp.pkl','rb') as f:
                traj = pickle.load(f)

            i = 0
            for waypoint in traj:
                if i % int(len(traj)/20) != 0:
                    i += 1
                    continue
                i += 1

                action = Base_pb2.Action()
                action.name = "Example angular action movement"
                action.application_data = ""
                # Execute Trajectory
                for joint_id in range(actuator_count.count):
                    joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
                    joint_angle.joint_identifier = joint_id
                    r_idx = waypoint['joints'].index(f'joint_{joint_id+1}')
                    joint_angle.value = np.rad2deg(waypoint['position'][r_idx]) % 360

                #e = threading.Event()
                #notification_handle = self.base_client.OnNotificationActionTopic(
                #    check_for_end_or_abort(e),
                #    Base_pb2.NotificationOptions()
                #)

                print("Executing action")
                self.base_client.ExecuteAction(action)

                print("Waiting for movement to finish ...")
                #finished = e.wait(TIMEOUT_DURATION)
                #self.base_client.Unsubscribe(notification_handle)
                time.sleep(0.05)
            """
            
            """
            print("Starting angular action movement ...")
            action = Base_pb2.Action()
            action.name = "Example angular action movement"
            action.application_data = ""

            actuator_count = self.base_client.GetActuatorCount()

            # Place arm straight up
            values_to_send = np.array([343.17868, 352.3781, 359.02692, 250.84962, 354.36536, 41.034626, 52.65804, 0.])
            idx = 0
            for joint_id in range(actuator_count.count):
                joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
                joint_angle.joint_identifier = joint_id
                joint_angle.value = values_to_send[idx]

                idx+=1

            e = threading.Event()
            notification_handle = self.base_client.OnNotificationActionTopic(
                check_for_end_or_abort(e),
                Base_pb2.NotificationOptions()
            )

            print("Executing action")
            self.base_client.ExecuteAction(action)

            print("Waiting for movement to finish ...")
            finished = e.wait(TIMEOUT_DURATION)
            self.base_client.Unsubscribe(notification_handle)
            """
            # END TEMP
            
            # Initialize command with zero efforts
            """
            for i in range(7):  # 7 joints for Gen3
                self.cyclic_command.actuators.add()
                #self.cyclic_command.actuators[i].torque_joint = 0.0
                self.cyclic_command.actuators[i].flags = 1  # servoing
            
            # Set actuator control mode
            for i in range(7):
                control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
                control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('POSITION')
                self.actuator_config.SetControlMode(control_mode_message, i+1) # IDs are +1 indexe

            self.get_logger().info("Set Control Mode")

            self.feedback = self.base_cyclic_client.Refresh(self.cyclic_command)

            self.get_logger().info("Sent Initial Arm Command")


            # TEMP: Test client
            # Send the command to the robot
            while True:
                self.base_client.ClearFaults()
                feedback = self.base_cyclic_client.Refresh(self.cyclic_command)
                command = np.zeros(8)
                for i in range(7):  # 7 joints for Gen3
                    #self.cyclic_command.actuators[i].flags = 1  # servoing
                    #self.cyclic_command.actuators[i].torque_joint = feedback.actuators[i].torque
                    #if i == 6:
                    #    self.cyclic_command.actuators[i].torque_joint = -10.0
                    #self.cyclic_command.actuators[i].flags = 1  # servoing
                    command[i]= feedback.actuators[i].position
                    if i == 1:
                        self.cyclic_command.actuators[i].position -= 0.1
                    #if i == 6 or i == 2:
                    #    self.cyclic_command.actuators[i].position += 0.1
            """
            # END TEMP

            self.kortex_connected = True
            self.get_logger().info("Successfully connected to Kinova robot!")
            self.publish_status("robot_connected")
            
        except Exception as e:
            self.get_logger().error(f"Failed to connect to robot: {e}")
            self.publish_status("robot_connection_failed")
            self.kortex_connected = False
    
    def joint_state_callback(self, msg: JointState):
        """Callback for joint state updates."""
        self.state_lock.acquire()
        # Update new data structure
        self.latest_data['joint_states'] = msg
        
        # Maintain backward compatibility
        if len(msg.position) >= 7:
            self.latest_joint_state = np.array(msg.position[:7])
        
        # Extract gripper position if available
        if len(msg.position) > 7:
            self.latest_gripper_position = np.array([msg.position[7]])
        else:
            self.latest_gripper_position = np.array([0.0])

        self.state_lock.release()
    
    def rgb_image_callback(self, msg: Image):
        """Callback for RGB camera image (primary from rosbag)."""
        self.state_lock.acquire()
        self.latest_data['rgb_image'] = msg
        # Also update legacy variable
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")
            resized = cv2.resize(cv_image, (self.img_width, self.img_height))
            self.latest_exterior_image = resized
        except Exception as e:
            self.get_logger().error(f"Error processing RGB image: {e}")

        self.state_lock.release()
    
    def exterior_image_callback(self, msg: Image):
        """Callback for exterior camera image (fallback)."""
        self.state_lock.acquire()
        # Only update if primary RGB image not available
        if self.latest_data['rgb_image'] is None:
            self.latest_data['rgb_image'] = msg
        
        # Update legacy variable
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")
            resized = cv2.resize(cv_image, (self.img_width, self.img_height))
            self.latest_exterior_image = resized
        except Exception as e:
            self.get_logger().error(f"Error processing exterior image: {e}")

        self.state_lock.release()
    
    def wrist_image_callback(self, msg: Image):
        """Callback for wrist camera image."""
        self.state_lock.acquire()
        self.latest_data['wrist_image'] = msg
        
        # Update legacy variable
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")
            resized = cv2.resize(cv_image, (self.img_width, self.img_height))
            self.latest_wrist_image = resized
        except Exception as e:
            self.get_logger().error(f"Error processing wrist image: {e}")

        self.state_lock.release()
    
    def dynamic_joint_callback(self, msg: DynamicJointState):
        """Callback for dynamic joint states (velocities)."""
        self.state_lock.acquire()
        self.latest_data['dynamic_joint_states'] = msg
        self.state_lock.release()
    
    def controller_state_callback(self, msg: JointTrajectoryControllerState):
        """Callback for controller state (pose information)."""
        self.state_lock.acquire()
        self.latest_data['controller_state'] = msg
        self.state_lock.release()
    
    def tf_callback(self, msg: TFMessage):
        """Callback for TF data."""
        self.state_lock.acquire()
        self.latest_data['tf_data'] = msg
        self.state_lock.release()
    
    def prompt_callback(self, msg: String):
        """Callback for language prompt updates."""
        self.state_lock.acquire()
        self.current_prompt = msg.data
        self.state_lock.release()
        self.get_logger().info(f"Updated prompt: {msg.data}")

    def model_status_callback(self, msg: Bool):
        self.model_loaded = msg.data

    def model_output_callback(self, msg: String):
        print("Waiting for Lock")
        self.fresh_actions_lock.acquire()
        print("Got Model Output")
        self.fresh_actions = True
        
        data = str(msg.data).encode('ascii')
        data = base64.b64decode(data)
        data = pickle.loads(data)
        self.current_actions = data
        
        self.fresh_actions_lock.release()

    def arm_eep_callback(self, msg: Float32MultiArray):
        print("In Arm EEP Callback")
        action = Base_pb2.Action()
        action.application_data = ""

        eep = msg.data

        pose = action.reach_pose.target_pose
        pose.x = eep[0]
        pose.y = eep[1]
        pose.z = eep[2]
        pose.theta_x = eep[3]
        pose.theta_y = eep[4]
        pose.theta_z = eep[5]

        e = threading.Event()
        notification_handle = self.base_client.OnNotificationActionTopic(
            check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        self.base_client.ExecuteAction(action)

        finished = e.wait(TIMEOUT_DURATION)
        self.base_client.Unsubscribe(notification_handle)
    
    def gripper_pos_callback(self, msg: Float32):
        print("In Gripper Pos Callback")
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.finger_identifier = 1
        finger.value = np.clip(msg.data, 0.0, 1.0)
        self.base_client.SendGripperCommand(gripper_command)
    
    def estop_status_callback(self, msg: String):
        """Callback for e-stop status updates (expects 'true'/'false' at ~30Hz)."""
        self.estop_lock.acquire()
        current_time = time.time()
        
        # Update last received time
        self.last_estop_time = current_time
        
        # Check if e-stop is released (not active)
        if msg.data:
            if self.estop_active:
                self.estop_active = False
                self.get_logger().info("E-stop released - robot enabled")
                self.publish_status("estop_released")
        else:
            if not self.estop_active:
                self.estop_active = True
                self.get_logger().warning("E-stop activated")
                self.publish_status("estop_activated")
                self.emergency_stop_efforts()

        self.estop_lock.release()
    
    def start_estop_watchdog(self):
        """Start the e-stop watchdog thread."""
        self.estop_watchdog_thread = threading.Thread(
            target=self.estop_watchdog_loop,
            daemon=True,
            name="estop_watchdog"
        )
        #self.estop_watchdog_thread.start()
        #self.get_logger().info("E-stop watchdog thread started")
    
    def estop_watchdog_loop(self):
        """
        Watchdog loop that monitors e-stop status and enforces timeout.
        Runs at ~30Hz and ensures robot is in e-stop mode unless actively receiving 'true' signals.
        Allows epsilon tolerance down to ~28Hz to account for network jitter and timing variations.
        """
        while self.estop_watchdog_active and rclpy.ok():
            try:
                current_time = time.time()
                
                self.estop_lock.acquire()
                time_since_last_signal = current_time - self.last_estop_time
                
                # Check if we've exceeded the timeout
                if time_since_last_signal > self.estop_timeout:
                    if not self.estop_active:
                        self.estop_active = True
                        self.get_logger().warning(
                            f"E-stop watchdog timeout ({time_since_last_signal:.3f}s > {self.estop_timeout}s) - "
                            "activating emergency stop"
                        )
                        self.publish_status("estop_watchdog_timeout")
                        self.emergency_stop_efforts()

                self.estop_lock.release()
                
                # Sleep for ~30Hz operation (33ms)
                time.sleep(0.033)
                
            except Exception as e:
                self.get_logger().error(f"Error in e-stop watchdog: {e}")
                # On error, activate e-stop for safety
                self.estop_lock_acquire()
                if not self.estop_active:
                    self.estop_active = True
                    self.emergency_stop_efforts()
                self.estop_lock.release()
                time.sleep(0.1)  # Longer sleep on error
    
    def is_estop_active(self) -> bool:
        """Check if e-stop is currently active."""
        self.estop_lock.acquire()
        ret_val = self.estop_active
        self.estop_lock.release()
        return ret_val
    
    def __del__(self):
        """Destructor - ensure safe shutdown."""
        try:
            # Stop watchdog thread
            if hasattr(self, 'estop_watchdog_active'):
                self.estop_watchdog_active = False
            
            if hasattr(self, 'kortex_connected') and self.kortex_connected:
                self.get_logger().info("Shutting down - sending zero efforts")
                self.emergency_stop_efforts()
                
            if hasattr(self, 'session_manager') and self.session_manager is not None:
                self.session_manager.CloseSession()
                self.session_manager_udp.CloseSession()
                
            if hasattr(self, 'transport') and self.transport is not None:
                self.transport.disconnect()
                self.transport_udp.disconnect()
        except Exception as e:
            # Avoid throwing exceptions in destructor
            print(f"Error in destructor: {e}")
    
    def get_observation(self) -> Optional[Dict]:
        """Construct observation dictionary for OpenPI (DROID format)."""
        self.state_lock.acquire()
        # Check for required data using new data structure
        required_keys = ['rgb_image', 'joint_states']
        if not all(self.latest_data[k] is not None for k in required_keys):
            return None
        
        # Use new data structure for enhanced processing
        obs_data = {
            'rgb_image': self.latest_data['rgb_image'],
            'wrist_image': self.latest_data['wrist_image'] or self.latest_data['rgb_image'],  # Fallback
            'joint_states': self.latest_data['joint_states'],
            'dynamic_joint_states': self.latest_data['dynamic_joint_states'],
            'controller_state': self.latest_data['controller_state'],
            'tf_data': self.latest_data['tf_data'],
        }

        self.state_lock.release()
        
        return obs_data
    
    def interpret_openpi_action(self, action_chunk: np.ndarray) -> Dict:
        """Interpret OpenPI action and convert to robot commands."""
        if action_chunk.shape[0] == 0:
            return {}
        
        # Take the first action from the chunk
        action = action_chunk[0]
        
        result = {}
        
        if self.action_mode == 'effort' and len(action) >= 7:
            # Direct effort/torque control
            result['target_efforts'] = [action[:6]]
            #result['target_efforts'] = action_chunk[:, :6]
            
        elif self.action_mode == 'joint' and len(action) >= 7:
            # Joint space control
            result['target_joints'] = action[:7].tolist()
            
        elif self.action_mode == 'pose' and len(action) >= 6:
            # Cartesian space control (assuming [x, y, z, rx, ry, rz] format)
            pose = Pose()
            pose.position.x = float(action[0])
            pose.position.y = float(action[1])
            pose.position.z = float(action[2])
            
            # Convert rotation vector to quaternion (simplified)
            # TODO: Implement proper rotation conversion based on your action format
            pose.orientation.x = float(action[3])
            pose.orientation.y = float(action[4])
            pose.orientation.z = float(action[5])
            pose.orientation.w = 1.0  # Placeholder - implement proper conversion
            
            result['target_pose'] = pose
        

        result['gripper_position'] = [action[6]]
        #result['gripper_position'] = action_chunk[:,6]
        
        # Gripper command - handle different action lengths
        """
        if self.action_mode == 'effort' and len(action) > 7:
            result['gripper_position'] = float(action[7])
        elif len(action) > 7:
            result['gripper_position'] = float(action[7])
        elif len(action) > 6 and self.action_mode != 'effort':
            result['gripper_position'] = float(action[6])
        else:
            print("No gripper position? Action Length: ", len(action))
        """
        
        return result
    
    def control_loop(self):
        """Main control loop - gets observations, runs model inference, plans and executes actions."""
        try:
            # Check e-stop status first - abort if e-stop is active
            if self.is_estop_active():
                self.get_logger().debug("E-stop is active - skipping control loop")
                print("E-stop is active - skipping control loop")
                return
            
            # Skip if model not loaded
            if not self.model_loaded:
                print(f"Skipping -- Model loaded: {self.model_loaded}")
                return
            
            # For effort mode, ensure robot is connected
            if self.action_mode == 'effort' and not self.kortex_connected:
                self.get_logger().debug("Robot not connected - cannot send effort commands")
                print("Robot not connected - cannot send effort commands")
                return
            
            # Get observation data using new approach
            obs_data = self.get_observation()
            if obs_data is None:
                self.get_logger().debug("Waiting for complete observation...")
                print("Waiting for complete observation...")
                return
            
            # Process data for model inference using DataProcessor
            model_input = self.data_processor.transform_for_inference(obs_data, self.current_prompt)


            # TEMP: Use end effector pose
            input_joint_angles = self.base_client.GetMeasuredJointAngles()
            e_p = self.base_client.ComputeForwardKinematics(input_joint_angles)

            gripper_request = Base_pb2.GripperRequest()

            # Wait for reported position to be opened
            gripper_request.mode = Base_pb2.GRIPPER_POSITION
            gripper_pos_b = self.base_client.GetMeasuredGripperMovement(gripper_request)
            for g in gripper_pos_b.finger:
                gripper_pos = g.value


            model_input['state'] = np.array([e_p.x, e_p.y, e_p.z, e_p.theta_x, e_p.theta_y, e_p.theta_z, gripper_pos])
            # END TEMP

            if model_input is None:
                self.get_logger().debug("Failed to process observation data")
                print("Failed to process observation data")
                return

            # Return if default prompt
            if self.current_prompt == self.default_prompt:
                self.get_logger().info("Skipping -- No new prompt set")
                return
            
            # Use legacy observation format
            print("Starting action sending")
            self.publish_model_input(model_input)

            # Wait for fresh actions
            while True:
                self.fresh_actions_lock.acquire()
                if self.fresh_actions:
                    self.fresh_actions = False
                    self.fresh_actions_lock.release()
                    break
                self.fresh_actions_lock.release()
                self.publish_model_input(model_input)
                time.sleep(0.1)

            result = self.current_actions

            
            if "actions" in result:
                action_chunk = result["actions"]
                
                # Interpret action
                commands = self.interpret_openpi_action(action_chunk)
                if commands:
                    # Handle different action modes
                    if 'target_efforts' in commands:
                        # Direct effort control
                        success = self.send_effort_command(commands['target_efforts'])
                        if success:
                            self.get_logger().debug(f"Sent effort command: {commands['target_efforts']}")
                        
                    elif 'target_pose' in commands:
                        self.get_logger().warning("Pose control mode not implemented without MoveIt")
                        
                    elif 'target_joints' in commands:
                        self.get_logger().warning("Joint control mode not implemented without MoveIt")
                    
                    # Send gripper command
                    if 'gripper_position' in commands:
                        self.send_gripper_command(commands['gripper_position'])
            
        except Exception as e:
            self.get_logger().error(f"Error in control loop: {e}")
            traceback.print_exc()
            self.publish_status("control_error")
    
    def send_effort_command(self, efforts: np.ndarray):
        """Send effort/torque commands directly to the robot via Kortex API."""
        # Check e-stop status before sending any command
        if self.is_estop_active():
            self.get_logger().debug("E-stop active - sending zero efforts")
            efforts = np.zeros(7)
        
        if not self.kortex_connected or self.base_cyclic_client is None:
            self.get_logger().warning("Robot not connected - cannot send effort commands")
            return False
        
        # Validate effort command
        #if not self.validate_effort_command(efforts):
        #    self.get_logger().error("Invalid effort command - sending zero efforts for safety")
        #    efforts = np.zeros(7)
        
        try:
            self.effort_lock.acquire()
            # Apply effort limits for safety

            # TEMP: DO NOT CLIP EFFORTS?
            limited_efforts = efforts

            #limited_efforts = limited_efforts / np.max(np.abs(limited_efforts)) # SCALE?
            #limited_efforts = np.rad2deg(limited_efforts) % 360
            # END TEMP
            
            # Update current efforts for monitoring
            self.current_efforts = limited_efforts.copy()
            
            # TEMP: REPEAT A FEW TIMES??
            action = Base_pb2.Action()
            action.name = "Example angular action movement"
            action.application_data = ""

            actuator_count = self.base_client.GetActuatorCount()

            cur_joints = self.base_client.GetMeasuredJointAngles()
            cur_angles = np.zeros(actuator_count.count)
            for joint_angle in cur_joints.joint_angles:
                cur_angles[joint_angle.joint_identifier] = joint_angle.value

            # Place arm straight up
            """
            i = 0
            for joint_id in range(actuator_count.count):
                joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
                joint_angle.joint_identifier = joint_id
                joint_angle.value = self.current_efforts[i]
                #joint_angle.value = self.current_efforts[i] + cur_angles[i]
                #ang = (self.current_efforts[i] - cur_angles[i]) / 50
                #ang += cur_angles[i]

                #joint_angle.value = ang % 360

                i += 1
            """
            
            for i in range(len(self.current_efforts)):
                # Publish
                msg = Float32MultiArray()
                msg.data = self.current_efforts[i].tolist()
                self.arm_eep_pub.publish(msg)
            # END TEMP


            # Publish current efforts for monitoring
            #self.publish_current_efforts(limited_efforts)

            self.effort_lock.release()
            
            return True
                
        except Exception as e:
            self.get_logger().error(f"Error sending effort command: {e}")
            # Attempt emergency stop on error
            self.emergency_stop_efforts()
            return False
    
    def publish_current_efforts(self, efforts: np.ndarray):
        """Publish current effort commands for monitoring."""
        msg = Float32MultiArray()
        msg.data = efforts.tolist()
        self.effort_pub.publish(msg)
    
    def emergency_stop_efforts(self):
        """Send zero efforts to all actuators immediately."""
        if self.kortex_connected and self.base_cyclic_client is not None:
            try:
                zero_efforts = np.zeros(7)
                self.effort_lock.acquire()
                for i in range(7):
                    if i < len(self.cyclic_command.actuators):
                        self.cyclic_command.actuators[i].torque_joint = 0.0
                self.base_cyclic_client.refresh_command(self.cyclic_command)
                self.current_efforts = zero_efforts

                self.effort_lock.release()

                self.get_logger().info("Emergency stop: All efforts set to zero")
                return True
            except Exception as e:
                self.get_logger().error(f"Failed to send emergency stop: {e}")
                return False
        return False
    
    def validate_effort_command(self, efforts: np.ndarray) -> bool:
        """Validate effort command for safety."""
        if len(efforts) != 7:
            self.get_logger().error(f"Invalid effort array length: {len(efforts)} (expected 7)")
            return False
        
        # Check for NaN or infinite values
        if not np.all(np.isfinite(efforts)):
            self.get_logger().error("Effort command contains NaN or infinite values")
            return False
        
        # Check effort limits
        max_effort = np.max(np.abs(efforts))
        if max_effort > self.effort_limit:
            self.get_logger().warning(f"Effort command exceeds limit: {max_effort} > {self.effort_limit}")
            # We'll clip the values in send_effort_command, so this is just a warning
        
        return True
    
    def send_gripper_command(self, position: float):
        """Send gripper command."""
        """
        gripper_cmd = GripperCommand()
        gripper_cmd.position = position
        gripper_cmd.max_effort = 100.0
        print("\n\nGOT TO SEND GRIPPER COMMAND\n\n")
        assert False
        self.gripper_pub.publish(gripper_cmd)
        """
        
        for i in range(len(position)):
            msg = Float32()
            msg.data = position[i]
            self.gripper_pos_pub.publish(msg)

    def publish_status(self, status: str):
        """Publish current status."""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
    
    def publish_plan_status(self, status: str):
        """Publish planning status."""
        msg = String()
        msg.data = status
        self.plan_status_pub.publish(msg)

    def publish_model_input(self, model_input):
        msg = String()
        data = pickle.dumps(model_input)
        data = base64.b64encode(data).decode('ascii')
        msg.data = data 
        self.model_input_pub.publish(msg)
    

def main(args=None):
    rclpy.init(args=args)
    
    node = None
    try:
        bridge_node = OpenPIKinovaDirectBridge()
        executor = MultiThreadedExecutor()
        executor.add_node(bridge_node)
        executor.spin()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received - performing safe shutdown...")
        if node is not None:
            # Stop watchdog thread
            node.estop_watchdog_active = False
            if node.action_mode == 'effort':
                print("Sending zero efforts before shutdown...")
                node.emergency_stop_efforts()
                time.sleep(0.1)  # Give time for command to be sent

            # Disconnect all comms with the arm
            node.sessionManager.CloseSession()
            node.sessionManager_udp.CloseSession()
            node.transport.disconnect()
            node.transport_udp.disconnect()
    finally:
        if node is not None:
            # Ensure watchdog is stopped
            if hasattr(node, 'estop_watchdog_active'):
                node.estop_watchdog_active = False
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

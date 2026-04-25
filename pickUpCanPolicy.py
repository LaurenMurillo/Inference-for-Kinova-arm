from __future__ import annotations

import time
import threading
import numpy as np
from typing import Optional
from PIL import Image as PILImage

# Kinova SDK imports
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Session_pb2
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport

# OpenPI imports
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config


class PickUpCanInference:
    """
    Inference class for the pi0.5 pick-up-can policy on a Kinova Gen3 arm.

    Main loop:
      1. Read the latest observations (cameras + end effector pose + gripper).
      2. Package them into an observation dict.
      3. Ask the policy for an action chunk of shape (10, 8).
      4. Execute one action row at a time at 10Hz.
      5. When the chunk is finished, read fresh observations and repeat.

    Action format (from training data info.json):
      action[0:6] = end effector delta pose [x, y, z, theta_x, theta_y, theta_z]
      action[6]   = gripper position [0.0 = open, 1.0 = closed]

    State format (what goes INTO the model):
      state[0:6] = current end effector pose [x, y, z, theta_x, theta_y, theta_z]
      state[6]   = current gripper position
    """

    def __init__(
        self,
        horizon: int = 10,           # from config: action_horizon=10
        action_dim: int = 8,         # from info.json: actions shape [8] (but we use 7)
        control_hz: float = 10.0,    # from info.json: fps=10
        task_prompt: str = "Grab the can",  # from test_model.py
        robot_ip: str = "192.168.1.10",     # from openpi_kinova_direct_bridge.py
        checkpoint_dir: str = "/home/ros/pytorch_models/openpi/cobot_finetune_grab_can_150_eep_with_gripper__004/29999",
        openpi_config: str = "pi05_cobot",  # from config.py
    ) -> None:

        self.horizon = horizon
        self.action_dim = action_dim
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz
        self.task_prompt = task_prompt
        self.robot_ip = robot_ip

        # ── Latest sensed observations ────────────────────────────────────────
        # These get updated every control step by reading directly from the arm.
        # Shape: (7,) — [x, y, z, theta_x, theta_y, theta_z, gripper]
        self.latest_state: Optional[np.ndarray] = None

        # Shape: (256, 256, 3) uint8 RGB — from external camera
        self.latest_ext_image: Optional[np.ndarray] = None

        # Shape: (256, 256, 3) uint8 RGB — from wrist camera
        self.latest_wrist_image: Optional[np.ndarray] = None

        # ── Chunk execution state ─────────────────────────────────────────────
        # The current chunk of actions returned by the policy.
        # Shape: (horizon, action_dim) = (10, 8)
        self.current_chunk: Optional[np.ndarray] = None

        # Which row of the current chunk to execute next.
        self.chunk_step: int = 0

        # ── Control flags ─────────────────────────────────────────────────────
        self.stop_requested = False

        # ── Connect to the Kinova arm ─────────────────────────────────────────
        print(f"Connecting to Kinova arm at {robot_ip}...")
        self.transport = TCPTransport()
        self.transport_udp = UDPTransport()
        self.transport.connect(robot_ip, 10000)
        self.transport_udp.connect(robot_ip, 10001)

        self.router = RouterClient(self.transport, lambda: None)
        self.router_udp = RouterClient(self.transport_udp, lambda: None)

        # Open a session (login to the arm)
        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = "admin"
        session_info.password = "admin"
        session_info.session_inactivity_timeout = 10000
        session_info.connection_inactivity_timeout = 2000

        self.session_manager = SessionManager(self.router)
        self.session_manager_udp = SessionManager(self.router_udp)
        self.session_manager.CreateSession(session_info)
        self.session_manager_udp.CreateSession(session_info)

        # base_client: for high level commands (get pose, send pose, gripper)
        # base_cyclic_client: for fast real-time feedback
        self.base_client = BaseClient(self.router)
        self.base_cyclic_client = BaseCyclicClient(self.router_udp)
        print("Connected to arm.")

        # ── Load the pi0.5 policy ─────────────────────────────────────────────
        print(f"Loading policy from {checkpoint_dir}...")
        train_config = _config.get_config(openpi_config)
        self.policy = _policy_config.create_trained_policy(
            train_config=train_config,
            checkpoint_dir=checkpoint_dir,
            pytorch_device="cuda",
        )
        print("Policy loaded.")

    # =========================================================================
    # Observation methods
    # =========================================================================

    def read_robot_state(self) -> np.ndarray:
        """
        Read the current end effector pose and gripper position directly
        from the arm.

        Returns shape (7,):
          [x, y, z, theta_x, theta_y, theta_z, gripper_position]

        This is what goes into the "state" key of the observation dict.
        We read this live from the arm every step so it reflects the actual
        current state, not what we last commanded.
        """
        # Get current joint angles and compute end effector pose
        joint_angles = self.base_client.GetMeasuredJointAngles()
        eep = self.base_client.ComputeForwardKinematics(joint_angles)

        # Get current gripper position (0.0 = open, 1.0 = closed)
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        gripper_feedback = self.base_client.GetMeasuredGripperMovement(gripper_request)
        gripper_pos = 0.0
        for finger in gripper_feedback.finger:
            gripper_pos = finger.value

        state = np.array(
            [eep.x, eep.y, eep.z, eep.theta_x, eep.theta_y, eep.theta_z, gripper_pos],
            dtype=np.float32,
        )
        return state

    def update_ext_image(self, image: np.ndarray) -> None:
        """
        Save the latest external camera RGB image.
        Expected shape: (256, 256, 3) uint8.

        In the ROS version this would be called from a camera subscriber callback.
        For now can call this manually before running the loop.
        """
        self.latest_ext_image = image

    def update_wrist_image(self, image: np.ndarray) -> None:
        """
        Save the latest wrist camera RGB image.
        Expected shape: (256, 256, 3) uint8.

        Same as above — call from a camera callback or manually.
        """
        self.latest_wrist_image = image

    def observations_ready(self) -> bool:
        """
        Return True only when all required observation inputs are available.
        Prevents inference from running before the first camera frames arrive.
        """
        return (
            self.latest_ext_image is not None
            and self.latest_wrist_image is not None
        )
        # Note: we don't check latest_state here because we read it live
        # inside build_observation_dict() every time.

    def build_observation_dict(self) -> dict:
        """
        Package the latest observations into the exact format the policy expects.

        Key names match what was used in training (confirmed from test_model.py
        and cobot_policy.py). Images are converted to PIL format because that
        is what test_model.py used when calling policy.infer().

        Returns a dict with keys:
          "image"       — external camera, PIL Image, 256x256 RGB
          "wrist_image" — wrist camera, PIL Image, 256x256 RGB
          "state"       — end effector pose + gripper, np.float32 shape (7,)
          "prompt"      — task instruction string
        """
        if not self.observations_ready():
            raise RuntimeError("Cannot build observation dict: camera images not ready.")

        # Read live robot state (end effector pose + gripper)
        state = self.read_robot_state()

        # Convert numpy images to PIL — matches how test_model.py fed data to policy
        ext_pil = PILImage.fromarray(self.latest_ext_image)
        wrist_pil = PILImage.fromarray(self.latest_wrist_image)

        obs = {
            "image":       ext_pil,       # external camera
            "wrist_image": wrist_pil,     # wrist camera
            "state":       state,         # current end effector pose + gripper
            "prompt":      self.task_prompt,
        }
        return obs

    # =========================================================================
    # Chunk creation — METHOD 3
    # =========================================================================

    def request_action_chunk(self, obs: dict) -> np.ndarray:
        """
        Send the observation dict to the policy and get back an action chunk.

        The policy returns a dict with key "actions" containing a numpy array
        of shape (horizon, action_dim) = (10, 8).

        Each row is one full low-level action for one timestep:
          row[0:6] = delta end effector pose [x, y, z, theta_x, theta_y, theta_z]
          row[6]   = gripper position [0.0=open, 1.0=closed]

        This matches how test_model.py called the policy:
          action_chunk = policy.infer(example)["actions"]
        """
        result = self.policy.infer(obs)
        chunk = np.asarray(result["actions"], dtype=np.float32)
        print(f"Got chunk of shape {chunk.shape}")
        return chunk

    def load_new_chunk(self) -> None:
        """
        Build a fresh observation dict, request a new action chunk from the
        policy, and reset the chunk step counter to zero.

        Called when there is no active chunk or the current one is finished.
        """
        obs = self.build_observation_dict()
        chunk = self.request_action_chunk(obs)

        if chunk.shape != (self.horizon, self.action_dim):
            raise ValueError(
                f"Unexpected chunk shape. Expected {(self.horizon, self.action_dim)}, "
                f"got {chunk.shape}."
            )

        self.current_chunk = chunk
        self.chunk_step = 0

    def chunk_finished(self) -> bool:
        """Return True if there is no chunk or we have executed all rows."""
        return (
            self.current_chunk is None
            or self.chunk_step >= len(self.current_chunk)
        )

    # =========================================================================
    # Action execution — METHODS 1 and 2
    # =========================================================================

    def send_arm_command(self, arm_command: np.ndarray) -> None:
        """
        Send one end effector pose command to the Kinova arm.

        arm_command shape: (6,) = [x, y, z, theta_x, theta_y, theta_z]

        The training config had extra_delta_transform=True, meaning the model
        was trained on DELTA actions (changes relative to current pose).
        So we add the delta to the current pose to get the target pose.

        The arm executes this as a Cartesian reach_pose action — it figures
        out the joint angles itself via built-in inverse kinematics.
        """
        # Read current end effector pose
        joint_angles = self.base_client.GetMeasuredJointAngles()
        current_eep = self.base_client.ComputeForwardKinematics(joint_angles)

        # Add the delta to get the target pose
        target_x       = current_eep.x       + float(arm_command[0])
        target_y       = current_eep.y       + float(arm_command[1])
        target_z       = current_eep.z       + float(arm_command[2])
        target_theta_x = current_eep.theta_x + float(arm_command[3])
        target_theta_y = current_eep.theta_y + float(arm_command[4])
        target_theta_z = current_eep.theta_z + float(arm_command[5])

        # Build the Kinova action message
        action = Base_pb2.Action()
        action.application_data = ""

        pose = action.reach_pose.target_pose
        pose.x       = target_x
        pose.y       = target_y
        pose.z       = target_z
        pose.theta_x = target_theta_x
        pose.theta_y = target_theta_y
        pose.theta_z = target_theta_z

        # Send — non-blocking so the control loop stays at 10Hz
        self.base_client.ExecuteAction(action)

    def send_gripper_command(self, gripper_position: float) -> None:
        """
        Send a gripper position command to the Kinova arm.

        gripper_position: float between 0.0 (fully open) and 1.0 (fully closed)

        Uses the Kinova high-level gripper position API — same method used
        in openpi_kinova_direct_bridge.py.
        """
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.finger_identifier = 1
        finger.value = float(np.clip(gripper_position, 0.0, 1.0))
        self.base_client.SendGripperCommand(gripper_command)

    def execute_action_row(self, action_row: np.ndarray) -> None:
        """
        Execute one row of the action chunk.

        Splits the row into arm command and gripper command, then
        sends both to the robot.

        action_row shape: (8,)
          action_row[0:6] = delta end effector pose
          action_row[6]   = gripper position
          action_row[7]   = padding (ignore — from cobot_policy.py's 8-dim output)
        """
        arm_command    = action_row[0:6]   # delta end effector pose
        gripper_command = float(action_row[6])  # gripper open/close

        self.send_arm_command(arm_command)
        self.send_gripper_command(gripper_command)

    # =========================================================================
    # Main control loop
    # =========================================================================

    def control_step(self) -> None:
        """
        Run one step of the inference and execution loop.

          1. Check observations are ready.
          2. If chunk is done (or missing), request a new one from the policy.
          3. Execute the next action row.
          4. Advance the chunk step counter.
        """
        if not self.observations_ready():
            print("Waiting for camera images...")
            return

        if self.chunk_finished():
            print("Chunk finished — requesting new chunk from policy...")
            self.load_new_chunk()

        action_row = self.current_chunk[self.chunk_step]
        self.execute_action_row(action_row)
        self.chunk_step += 1

    def run(self) -> None:
        """
        Main loop — runs continuously at control_hz until stop() is called.

        Each iteration:
          - calls control_step() which handles observation + inference + execution
          - sleeps for the remaining time in the control interval
        """
        print(f"Starting inference loop at {self.control_hz}Hz. Press Ctrl+C to stop.")
        while not self.stop_requested:
            t_start = time.time()

            self.control_step()

            # Sleep for however long is left in this control interval
            elapsed = time.time() - t_start
            sleep_time = max(0.0, self.dt - elapsed)
            time.sleep(sleep_time)

    def stop(self) -> None:
        """
        Request the loop to stop cleanly after the current step finishes.
        """
        self.stop_requested = True
        print("Stop requested.")

    def disconnect(self) -> None:
        """
        Close the connection to the arm cleanly.
        Always call this when you are done.
        """
        self.session_manager.CloseSession()
        self.session_manager_udp.CloseSession()
        self.transport.disconnect()
        self.transport_udp.disconnect()
        print("Disconnected from arm.")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    inference = PickUpCanInference()

    try:
        inference.run()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        inference.stop()
        inference.disconnect()
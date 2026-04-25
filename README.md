# Inference-for-Kinova-arm
inference for pick-up-a-can policy on kinova arm
This repo has two files, one that makes very slight modifications to an existing inference code file from the [https://github.com/ut-amrl/openpi_kinova_ros2/blob/main/scripts/openpi_kinova_direct_bridge.py] (ut-amrl/ openpi_kinova_ros2 repo) -> openpi_kinova_direct_bridge.py file. And another more custom inference code file that focusses on the action chunking functionality, using "dummy" data to prove this process would work when properly set up to take in actual model output data.

---

# Pick Up Can Policy Inference Deployment Guide

## Overview

This guide explains how to deploy the trained `pi05_cobot` pick-up-can policy on the Kinova Gen3 7-DOF arm using the `openpi_kinova_direct_bridge.py` inference node.

The system uses two ROS2 nodes that communicate with each other:

```
[model_node.py]                        [openpi_kinova_direct_bridge.py]
────────────────                       ────────────────────────────────
Loads the pi0.5 policy onto GPU        Reads cameras + arm state
Runs inference when asked              Builds observation dict
Returns action chunks                  Sends arm + gripper commands
        │                                          │
        └──── communicate via ROS2 topics ─────────┘
```

---

## Prerequisites

Before running inference, confirm the following:

- Kinova Gen3 7-DOF arm is powered on and connected via ethernet
- Arm is reachable at `192.168.1.10`
- Both cameras are plugged in and powered (external + wrist)
- The trained checkpoint exists at:
```
/home/ros/pytorch_models/openpi/cobot_finetune_grab_can_150_eep_with_gripper__004/29999
```
- You have SSH access to the robot machine at `10.1.0.13`

---

## Setup

### Step 1: SSH into the robot machine

```bash
ssh ros@10.1.0.13
```

### Step 2: Source ROS2 and the workspace

Run these two lines in every terminal you open before running any ROS2 commands:

```bash
source /opt/ros/humble/setup.bash
source /home/ros/cobot_fri/install/setup.bash
```

> **Note:** If `ros2: command not found` appears, check `cat ~/.bashrc` for the correct ROS2 source path on this machine.

### Step 3: Build the package (only needed if files were changed)

```bash
cd /home/ros/cobot_fri
colcon build --packages-select openpi_kinova_ros2
source install/setup.bash
```

---

## Running Inference

You need **four terminals** all SSHed into the robot machine. Open each one with:
```bash
ssh ros@10.1.0.13
```
And source ROS2 in each:
```bash
source /opt/ros/humble/setup.bash
source /home/ros/cobot_fri/install/setup.bash
```

---

### Terminal 1: Start the cameras and sensors

```bash
cd /home/ros/cobot_fri/tmux/pi_record
tmuxinator start
```

Wait about 10 seconds for all camera nodes to finish starting before continuing.

---

### Terminal 2: Start the model node

This loads the pi0.5 policy onto the GPU and waits for observations to run inference on.

```bash
ros2 run openpi_kinova_ros2 model_node \
    --ros-args \
    -p openpi_config:=pi05_cobot \
    -p openpi_checkpoint_dir:=/home/ros/pytorch_models/openpi/cobot_finetune_grab_can_150_eep_with_gripper__004/29999 \
    -p pytorch_device:=cuda
```

Wait until you see this in the terminal before continuing:
```
Model Status: True
```

---

### Terminal 3: Start the bridge node

This connects to the arm, reads camera images and end effector state, and executes the action chunks returned by the model node.

```bash
ros2 run openpi_kinova_ros2 openpi_kinova_direct_bridge \
    --ros-args \
    -p robot_ip:=192.168.1.10 \
    -p robot_port:=10000 \
    -p openpi_config:=pi05_cobot \
    -p openpi_checkpoint_dir:=/home/ros/pytorch_models/openpi/cobot_finetune_grab_can_150_eep_with_gripper__004/29999 \
    -p action_mode:=pose \
    -p control_frequency:=10.0 \
    -p image_width:=256 \
    -p image_height:=256 \
    -p default_prompt:="Do nothing"
```

Wait until you see this before continuing:
```
Successfully connected to Kinova robot!
```

---

### Terminal 4: Enable the e-stop and send the task prompt

The bridge node starts in e-stop mode for safety. The robot will not move until it receives a continuous e-stop release signal. Run this first to enable motion:

```bash
ros2 topic pub /arm_estop std_msgs/msg/Bool "data: true" --rate 30
```

> **Keep this running in the background.** If it stops publishing, the watchdog will automatically halt the robot.

Then in a **fifth terminal**, send the task prompt to start execution:

```bash
ros2 topic pub /openpi/prompt std_msgs/msg/String "data: 'Grab the can'" --once
```

The arm will begin moving immediately after receiving the prompt.

---

## What You Should See

Once everything is running correctly, Terminal 3 output should look like:

```
Successfully connected to Kinova robot!
Got Current EE Pose
Starting action sending
Got Model Output
[arm moves]
Got Current EE Pose
Starting action sending
Got Model Output
[arm moves]
... repeats until can is picked up
```

---

## Physical Setup

- Place the coke can in the same position relative to the arm as during data collection
- Make sure nothing is in the arm's path as it will begin moving immediately after the prompt is sent
- Keep a hand near the physical e-stop button at all times during the first test runs

---

## How to Stop Safely

Press `Ctrl + C` in Terminal 3 (the bridge node) first. Then press `Ctrl + C` in Terminal 4 to stop the e-stop publisher. The bridge's shutdown code will close all arm sessions cleanly.

> **Never just close the terminal.** Always use `Ctrl + C` so the Kortex API session closes properly.

If the arm behaves unexpectedly at any point, press the **physical e-stop button** on the robot first, then `Ctrl + C`.

---

## Monitoring

While the system is running you can monitor it in additional terminals:

```bash
# Check if the model loaded successfully
ros2 topic echo /openpi/model_status

# Check general bridge status
ros2 topic echo /openpi/status

# Check current effort/pose commands being sent
ros2 topic echo /openpi/current_efforts
```

---

## Configuration Reference

| Parameter | Value | Description |
|---|---|---|
| `openpi_config` | `pi05_cobot` | Training config name |
| `openpi_checkpoint_dir` | `/home/ros/pytorch_models/openpi/cobot_finetune_grab_can_150_eep_with_gripper__004/29999` | Trained model checkpoint |
| `robot_ip` | `192.168.1.10` | Kinova arm IP address |
| `robot_port` | `10000` | Kinova TCP port |
| `action_mode` | `pose` | End effector Cartesian control |
| `control_frequency` | `10.0` | Control loop Hz |
| `image_width` / `image_height` | `256` | Camera image size (must match training) |
| `default_prompt` | `Do nothing` | Robot stays still until prompt is changed |
| `estop_topic` | `/arm_estop` | Topic to publish e-stop release signal |
| `estop_timeout` | `0.045` | Seconds before watchdog triggers (45ms) |

---

## Troubleshooting

| Problem | What to check |
|---|---|
| `ros2: command not found` | Run `source /opt/ros/humble/setup.bash` first |
| `Failed to connect to robot` | Check ethernet cable, arm powered on, ping `192.168.1.10` |
| `Model Status: False` | Check checkpoint path exists, CUDA available (`nvidia-smi`) |
| `Waiting for complete observation` | Camera topics not publishing, check tmuxinator started correctly |
| `E-stop is active: skipping control loop` | Make sure Terminal 4 is publishing to `/arm_estop` at 30Hz |
| `Arm moves but ignores the can` | Check coke can is in same position as during data collection |
| `Chunk shape mismatch` error | Action horizon or action dim changed, check `pi05_cobot` config |

---

## Key Files

| File | Location | Purpose |
|---|---|---|
| Bridge node | `src/openpi_kinova_ros2/scripts/openpi_kinova_direct_bridge.py` | Main inference + arm control |
| Model node | `src/openpi_kinova_ros2/scripts/model_node.py` | Loads policy and runs inference |
| Training config | `external/openpi/src/openpi/training/config.py` | `pi05_cobot` config block |
| Cobot policy | `external/openpi/src/openpi/policies/cobot_policy.py` | Observation/action format definition |
| Dataset info | `lerobot_dataset_.../meta/info.json` | Confirms image size, action dim, fps |

---

## Training Details (for reference)

| Detail | Value | Source |
|---|---|---|
| Model | π0.5 (`pi05_cobot`) | `config.py` |
| Action horizon | 10 actions per chunk | `config.py` |
| Action format | Delta end effector pose `[x, y, z, θx, θy, θz, gripper]` | `info.json` + `cobot_policy.py` |
| State format | Current end effector pose `[x, y, z, θx, θy, θz, gripper]` | `openpi_kinova_direct_bridge.py` |
| Image size | 256 × 256 RGB | `info.json` |
| Control rate | 10 Hz | `info.json` fps field |
| Training episodes | 150 grab-can demonstrations | `info.json` |
| Task prompt | `"Grab the can"` | `test_model.py` |

# 7785 Autonomy Stack

A ROS 2 autonomy stack for TurtleBot3 that enables autonomous navigation through maze-like environments using LiDAR-based localization, sign recognition, and path planning.

## Overview

The stack combines two navigation packages to handle both local cell-level navigation and global multi-cell path planning:

- **sigi** (Sign-based Intelligent Grid-aware navigation) — Navigates within individual cells using LiDAR wall detection and reads directional signs with a MobileNetV3 classifier
- **wumbus** — Handles global navigation with A\* path planning, occupancy grid mapping, ICP-corrected odometry, and pure pursuit path tracking
- **sigi_interfaces** — Custom ROS 2 service definitions (`ReadSign.srv`)

## Architecture

```
sigi (cell-level)                    wumbus (global)
├── lidar.py        wall detection   ├── search_node.py       A* pathfinding
│                   & localization   ├── controller.py        pure pursuit tracking
├── controller.py   cell navigation  ├── odom_correct.py      ICP odometry correction
│                   state machine    ├── mapper.py            occupancy grid mapping
├── camera.py       sign recognition ├── derez_lidar.py       point cloud transform
│                   (MobileNetV3)    ├── transform_to_global.py  frame transform
└── geometry_utils.py                └── goal_tracker.py      goal management
```

### sigi

Handles single-cell navigation with a state machine that centers the robot, detects walls via iterative RANSAC, and reads signs on demand through a `/read_sign` service. Sign classification uses a MobileNetV3-Small model trained on Canny edge-detected images, recognizing 6 classes: `left`, `right`, `stop`, `do_not_enter`, `goal`, and `empty`.

### wumbus

Handles multi-cell autonomous navigation. Builds an occupancy grid from LiDAR scans, plans obstacle-free paths with A\*, and tracks them using pure pursuit control. Odometry is corrected by fusing wheel encoders with ICP scan matching to compensate for drift and wheel slip.

## Dependencies

- ROS 2
- Python 3
- PyTorch >= 2.0, torchvision >= 0.15, timm >= 1.0
- OpenCV >= 4.8
- NumPy, scikit-learn, Pillow

## Building

```bash
colcon build --packages-select sigi_interfaces sigi wumbus
source install/setup.bash
```

## Running

```bash
# Cell-level navigation
ros2 launch sigi navigation.launch.py

# Global navigation
ros2 launch wumbus wumbus.launch.py
```

## Sign Classifier

The `splatch/` directory contains the training pipeline for the sign classifier:

```bash
pip install -r splatch/requirements.txt
python3 splatch/train_mobilenetv4.py
python3 splatch/model_grader.py --data_path ./val_dataset --model_path ./mobilenetv4_sign_classifier.pth
```

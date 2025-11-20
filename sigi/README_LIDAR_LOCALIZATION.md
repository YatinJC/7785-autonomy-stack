# LiDAR-Based Cell Localization for Maze Navigation

## Overview

The enhanced `lidar.py` node provides precise within-cell localization for a TurtleBot3 navigating a 1x1m square-grid maze. It uses multi-wall RANSAC detection to determine both the robot's position and orientation within each cell, regardless of the robot's rotation.

## Key Features

### 1. Multi-Wall Detection
- Detects up to 3 walls simultaneously using iterative RANSAC
- Filters walls by distance (≤1m) to exclude walls from adjacent cells
- Clamps wall length to 1m maximum (cell size)
- Validates wall geometry (parallel/perpendicular constraints)

### 2. Rotation-Invariant Localization
- Works with arbitrary robot rotation (not limited to cardinal directions)
- Solves robot orientation by aligning detected wall angles to cardinal directions
- Uses wall distances to triangulate position within cell
- Provides confidence metric based on number of walls detected

### 3. Real-Time Visualization
- Color-coded wall markers (red, green, blue) in RViz
- Yellow cell boundary square showing estimated robot position
- Legacy single-wall visualization for backward compatibility

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cell_size` | 1.0m | Size of each maze cell |
| `max_walls` | 3 | Maximum number of walls to detect |
| `ransac_threshold` | 0.03m | Distance threshold for RANSAC inliers |
| `min_inliers` | 15 | Minimum points required per wall |
| `min_wall_length` | 0.3m | Minimum wall length to be considered valid |
| `angle_tolerance` | 5.0° | Tolerance for perpendicular/parallel checks |
| `max_wall_distance` | 1.0m | Maximum distance for walls (filters other cells) |

## Published Topics

### New Topics

**`/detected_walls`** (String - JSON)
```json
{
  "num_walls": 2,
  "walls": [
    {
      "id": 0,
      "slope_rad": -0.785,
      "slope_deg": -45.0,
      "distance_m": 0.25,
      "length_m": 0.8,
      "inliers": 42
    }
  ]
}
```

**`/cell_position`** (String - JSON)
```json
{
  "cell_x": 0.25,        // meters from left edge
  "cell_y": 0.4,         // meters from bottom edge
  "theta_rad": 0.785,    // robot orientation
  "theta_deg": 45.0,
  "confidence": 0.67,    // 0-1 (based on num_walls/3)
  "num_walls": 2
}
```

**`/wall_markers`** (visualization_msgs/MarkerArray)
- Visualizes all detected walls in different colors
- Frame: `base_scan`

**`/cell_boundary`** (visualization_msgs/Marker)
- Visualizes 1x1m cell boundary square (yellow)
- Square is aligned with MAZE axes (not robot axes)
- Rotates with the robot to show cell orientation
- Robot position is at the origin in base_link frame
- Frame: `base_link`

### Legacy Topics (Backward Compatibility)

- `/line_detection` - Single best wall detection
- `/line_marker` - Single wall visualization marker

## How It Works

### Algorithm Flow

1. **LiDAR Data Processing**
   - Convert polar to Cartesian coordinates
   - Filter valid ranges (≤2m)

2. **Multi-Wall Detection**
   - Iteratively apply RANSAC to find up to 3 walls
   - Remove inliers after each detection
   - Filter walls by distance (≤1m) and length (≥0.3m)
   - Clamp wall length to 1m max

3. **Geometry Validation**
   - Check all wall pairs are parallel or perpendicular
   - Validates maze constraints

4. **Orientation Solving**
   - Try aligning each detected wall to cardinal directions (0°, 90°, 180°, 270°)
   - Generate possible robot orientations
   - Use median as robust estimate

5. **Position Calculation**
   - Transform wall normals from robot frame to maze frame using `robot_theta`
   - Identify which cell edge each wall corresponds to (left/right/top/bottom)
   - Calculate position constraints:
     - Left wall (x=0): `robot_x = distance`
     - Right wall (x=1): `robot_x = 1.0 - distance`
     - Bottom wall (y=0): `robot_y = distance`
     - Top wall (y=1): `robot_y = 1.0 - distance`
   - Use median of estimates for robustness
   - Clamp to cell bounds (0-1m)

6. **Confidence Estimation**
   - More walls = higher confidence
   - 3 walls → confidence = 1.0
   - 2 walls → confidence = 0.67
   - 1 wall → insufficient for full localization

## Usage

### Running the Node

```bash
ros2 run sigi lidar
```

### With Custom Parameters

```bash
ros2 run sigi lidar --ros-args \
  -p cell_size:=1.0 \
  -p max_wall_distance:=1.0 \
  -p ransac_threshold:=0.02
```

### Visualizing in RViz

Add the following displays:

1. **MarkerArray** → `/wall_markers`
   - Shows detected walls in red, green, blue

2. **Marker** → `/cell_boundary`
   - Shows 1x1m yellow cell boundary

3. **Marker** → `/line_marker` (legacy)
   - Shows single best wall

### Monitoring Localization

```bash
# Watch cell position updates
ros2 topic echo /cell_position

# Watch detected walls
ros2 topic echo /detected_walls
```

## Coordinate Frames

- **`base_scan`**: LiDAR frame (where walls are detected)
- **`base_link`**: Robot frame (where cell boundary is drawn)
- **`maze`**: Virtual frame where cell is axis-aligned and walls are at 0°/90°/180°/270°

## Detailed Algorithm Explanation

### Position Fitting Process

The algorithm solves for robot position (x, y, θ) within a cell using geometric constraints from detected walls.

#### **Step 1: Detect Walls in Robot Frame**

LiDAR detects walls as lines in the robot's frame. Each wall is represented as:
```
ax + by + c = 0  (normalized, so |c| = perpendicular distance)
```
- `(a, b)` = wall normal direction (points toward robot)
- `|c|` = perpendicular distance from robot to wall

#### **Step 2: Solve for Robot Orientation**

Since maze walls are at cardinal directions (0°, 90°, 180°, 270°) in the maze frame, and we observe them at angle `α` in robot frame:
```
α = cardinal_direction - robot_theta

Therefore: robot_theta = cardinal_direction - α
```

For each detected wall, we try all 4 cardinal directions and collect possible `robot_theta` values. The median is used as the robust estimate.

**Example:**
- Wall detected at 45° in robot frame
- Could be: 0° - θ = 45° → θ = -45°
- Or: 90° - θ = 45° → θ = 45°
- Or: 180° - θ = 45° → θ = 135°
- Or: -90° - θ = 45° → θ = -135°
- With multiple walls, the correct θ will dominate

#### **Step 3: Transform Wall Normals to Maze Frame**

Once we know `robot_theta`, transform each wall's normal from robot frame to maze frame:
```
normal_maze = Rotation(robot_theta) * normal_robot

normal_maze_x = cos(θ) * a - sin(θ) * b
normal_maze_y = sin(θ) * a + cos(θ) * b
```

#### **Step 4: Identify Which Cell Wall**

In the maze frame, cell walls have specific normal directions:
- **Left wall** (x=0): normal = (+1, 0) [points right, into cell]
- **Right wall** (x=1): normal = (-1, 0) [points left, into cell]
- **Bottom wall** (y=0): normal = (0, +1) [points up, into cell]
- **Top wall** (y=1): normal = (0, -1) [points down, into cell]

Match each detected wall to the closest cardinal normal using dot product.

#### **Step 5: Calculate Position from Constraints**

Each identified wall gives a constraint:
```
Left wall:   robot is at distance d from x=0  →  x = d
Right wall:  robot is at distance d from x=1  →  x = 1 - d
Bottom wall: robot is at distance d from y=0  →  y = d
Top wall:    robot is at distance d from y=1  →  y = 1 - d
```

With 2+ walls, we have 2+ constraints. Use median for robustness against outliers.

**Example:**
```
Robot at 45° rotation, detects:
- Wall A: distance=0.3m, normal in robot frame = (-0.707, -0.707)
- Wall B: distance=0.4m, normal in robot frame = (0.707, -0.707)

Transform normals to maze frame (θ=45°):
- Wall A: normal_maze ≈ (-1, 0) → Right wall → x = 1 - 0.3 = 0.7
- Wall B: normal_maze ≈ (0, -1) → Top wall → y = 1 - 0.4 = 0.6

Position: (0.7, 0.6) at 45° rotation
```

### Why This Works

1. **Rotation invariant**: Works at any robot orientation
2. **Geometric constraints**: Uses actual perpendicular distances, not assumptions
3. **Robust**: Median handles measurement noise and outliers
4. **No prior map needed**: Assumes only that maze cells are square

## Expected Performance

- **Position Accuracy**: ±2-5cm within cell
- **Orientation Accuracy**: ±5° (depends on wall alignment)
- **Update Rate**: ~10Hz (LiDAR scan rate)
- **Minimum Requirements**: 2 walls for full localization

## Limitations & Future Improvements

### Current Limitations

1. **Position calculation is simplified**: Assumes walls can be classified as horizontal/vertical in robot frame
2. **No wall identification**: Doesn't track which specific wall (north/south/east/west) is detected
3. **Single cell assumption**: Doesn't handle multi-cell mapping

### Potential Improvements

1. **Bayesian position estimation**: Use particle filter or EKF for smoother estimates
2. **Wall identification**: Use map knowledge to identify specific walls
3. **Multi-cell tracking**: Integrate with global localization (AMCL)
4. **Corner detection**: Use corner features for more accurate positioning
5. **Temporal filtering**: Smooth position estimates over time

## Integration with Existing Stack

This node can be integrated with the existing `wumbus` autonomy stack:

- **`odom_correct.py`**: Can use `/cell_position` to snap odometry to known cell positions
- **`controller.py`**: Can use precise within-cell position for fine-grained control
- **`mapper.py`**: Can use wall detections to build more accurate maps

## Example Scenarios

### Scenario 1: Robot at 45° in corner
```
Detected walls: 2 (perpendicular)
Wall 1: distance=0.2m, angle=-45°
Wall 2: distance=0.3m, angle=+45°
→ Position: (0.2, 0.3), θ=45°, confidence=0.67
```

### Scenario 2: Robot in middle of hallway
```
Detected walls: 2 (parallel)
Wall 1 (left): distance=0.4m, angle=90°
Wall 2 (right): distance=0.6m, angle=-90°
→ Position: (0.5, 0.4), θ=0°, confidence=0.67
```

### Scenario 3: Robot in dead-end
```
Detected walls: 3
Front wall: distance=0.3m
Left wall: distance=0.2m
Right wall: distance=0.8m
→ Full localization with confidence=1.0
```

## Troubleshooting

**No walls detected**
- Check LiDAR is working: `ros2 topic echo /scan`
- Reduce `min_inliers` parameter
- Increase `ransac_threshold`

**Walls detected but geometry validation fails**
- Increase `angle_tolerance` parameter
- Check maze walls are actually perpendicular

**Position seems incorrect**
- Verify `cell_size` parameter matches actual maze
- Check `max_wall_distance` is set to cell size
- Watch `/wall_markers` in RViz to see what's detected

**Confidence always low**
- Robot may only see 1-2 walls
- Position robot to see more walls (e.g., in corner)
- Reduce `min_wall_length` to detect shorter wall segments

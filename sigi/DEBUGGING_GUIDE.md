# LiDAR Localization Debugging Guide

## What Was Added

I've enhanced the [lidar.py](sigi/lidar.py) node with comprehensive debugging features and improved orientation estimation to help diagnose and fix localization issues like the one shown in your RViz screenshot.

## New Features

### 1. **Improved Orientation Estimation (Lines 386-456)**

**Problem:** The original algorithm tried all possible cardinal alignments equally, leading to ambiguous results with only 2 walls.

**Solution:** Now uses perpendicular wall constraints when available:
- If two walls are perpendicular (90° apart), both must map to cardinals that are also 90° apart
- Tests both orderings: (wall1→0°, wall2→90°) and (wall1→90°, wall2→0°)
- Only accepts solutions where both walls agree on robot orientation (within 17° tolerance)
- Falls back to original method if no perpendicular pair found

**Example:**
```
Wall 1 at 45° in robot frame
Wall 2 at -45° in robot frame (perpendicular)

New algorithm tests:
- Are they at (0°, 90°)? → Robot at 45°
- Are they at (90°, 180°)? → Robot at -45°
- Are they at (180°, 270°)? → Robot at 135°
- etc.

Picks the orientation where both walls agree!
```

### 2. **Comprehensive Debug Logging**

#### **Wall Detection Logging (Lines 300-306)**
For each detected wall:
```
Detected wall 0: slope=45.0°, distance=0.300m, length=0.800m, normal=(0.707, 0.707)
Detected wall 1: slope=-45.0°, distance=0.400m, length=0.750m, normal=(-0.707, 0.707)
```

#### **Orientation Calculation Logging (Lines 425-456)**
Shows which method was used and the result:
```
Using perpendicular constraint: 2 candidates, theta=45.0°
Wall angles: [45.0, -45.0] deg, Robot theta: 45.0°
```

Or if using fallback:
```
No perpendicular pair, using all combinations: theta=45.0°
```

#### **Wall Classification Logging (Lines 521-529)**
For each wall after rotation to maze frame:
```
Wall identified: left, distance=0.300m, normal_maze=(1.00, 0.02), dot=0.98
Wall identified: bottom, distance=0.400m, normal_maze=(-0.01, 1.00), dot=0.99
```

Or if rejected:
```
Wall rejected: best_dot=0.45 < 0.7, normal_maze=(0.50, 0.50)
```

### 3. **Visual Debugging with Wall Normals (Lines 695-731)**

Added **yellow arrows** in RViz showing wall normal directions:
- Arrow starts at the closest point on each wall
- Arrow points toward the robot (origin)
- 20cm long, yellow color
- Published in `base_scan` frame
- Namespace: `wall_normals`

This helps verify:
- Normals are pointing the correct direction
- Wall detection is accurate
- RANSAC is fitting lines correctly

## How to Use for Debugging

### Step 1: Run the Node and Watch Logs

```bash
ros2 run sigi lidar
```

You should see output like:
```
[INFO] [lidar_processor]: Detected wall 0: slope=45.0°, distance=0.300m, ...
[INFO] [lidar_processor]: Detected wall 1: slope=-45.0°, distance=0.400m, ...
[INFO] [lidar_processor]: Using perpendicular constraint: 2 candidates, theta=45.0°
[INFO] [lidar_processor]: Wall angles: [45.0, -45.0] deg, Robot theta: 45.0°
[INFO] [lidar_processor]: Wall identified: left, distance=0.300m, normal_maze=(1.00, 0.02), dot=0.98
[INFO] [lidar_processor]: Wall identified: bottom, distance=0.400m, normal_maze=(-0.01, 1.00), dot=0.99
[INFO] [lidar_processor]: Cell position: (0.300, 0.400) m, theta: 45.0°, confidence: 0.67
[INFO] [lidar_processor]: Detected 2 walls
```

### Step 2: Check for Problems

#### **Problem: Wrong Theta**
```
[INFO] [lidar_processor]: Wall angles: [45.0, -45.0] deg, Robot theta: 135.0°
```
This means the perpendicular constraint picked the wrong solution. Check if:
- The walls are truly perpendicular
- The tolerance (0.3 rad = 17°) needs adjustment

#### **Problem: Walls Rejected**
```
[WARN] [lidar_processor]: Wall rejected: best_dot=0.45 < 0.7, normal_maze=(0.50, 0.50)
```
This means after rotating to maze frame, the wall normal doesn't align with any cardinal direction. Indicates wrong `robot_theta`.

#### **Problem: No Perpendicular Pair**
```
[INFO] [lidar_processor]: No perpendicular pair, using all combinations: theta=45.0°
```
The walls aren't perpendicular, so falling back to less accurate method.

### Step 3: Visualize in RViz

Add these displays to RViz:

1. **Detected Walls** (Red/Green/Blue lines)
   - Type: MarkerArray
   - Topic: `/wall_markers`
   - Shows the wall segments

2. **Wall Normals** (Yellow arrows) **← NEW**
   - Type: MarkerArray
   - Topic: `/wall_markers`
   - Namespace filter: `wall_normals`
   - Shows where normals point

3. **Cell Boundary** (Yellow square)
   - Type: Marker
   - Topic: `/cell_boundary`
   - Shows estimated cell edges

**What to Look For:**
- Yellow arrows should point from walls toward robot origin
- Cell boundary edges should align with detected walls
- If misaligned → check theta calculation in logs

### Step 4: Interpret the Results

#### **Good Localization:**
```
Logs:
  Wall angles: [45.0, -45.0] deg, Robot theta: 45.0°
  Wall identified: left, distance=0.300m, dot=0.98
  Wall identified: bottom, distance=0.400m, dot=0.99
  Cell position: (0.300, 0.400) m, theta: 45.0°

RViz:
  - Yellow square edges align with red/green wall lines
  - Yellow normal arrows point toward origin
  - Robot appears inside square at estimated position
```

#### **Bad Localization (Your Screenshot):**
```
Logs (expected):
  Wall angles: [some_angle, other_angle] deg, Robot theta: WRONG°
  Wall identified: WRONG_TYPE, distance=X, dot=LOW

RViz:
  - Yellow square rotated wrong relative to walls
  - Cell edges don't align with detected walls
  - Position estimate is off
```

## Tuning Parameters

If localization is still failing:

### **Angle Tolerance for Perpendicular Constraint (Line 408)**
```python
if abs(theta_from_i - theta_from_j) < 0.3:  # ~17 degrees
```
- **Increase** if rejecting valid solutions (too strict)
- **Decrease** if accepting wrong solutions (too loose)

### **Cardinal Matching Threshold (Line 514)**
```python
if best_match and best_dot > 0.7:
```
- **Lower** (e.g., 0.5) if walls are being rejected
- **Raise** (e.g., 0.85) if wrong walls are accepted

### **Perpendicular Detection Tolerance (Line 373 in `validate_wall_geometry`)**
```python
self.angle_tolerance = np.radians(5.0)  # degrees
```
- Set via parameter: `angle_tolerance`
- Increase if walls aren't being recognized as perpendicular

## Common Issues and Fixes

### Issue 1: Cell Rotated Wrong (Like Your Screenshot)

**Cause:** Orientation calculation picked wrong cardinal alignment

**Debug:**
1. Check logs for `Wall angles` and `Robot theta`
2. Manually calculate: if walls at 45° and -45°, robot should be at 45° or -45°
3. If theta is 90° off, the perpendicular constraint failed

**Fix:**
- Adjust tolerance in line 408
- Check if walls are truly perpendicular (validate_wall_geometry logs)

### Issue 2: Walls Not Identified

**Symptoms:**
```
[WARN] [lidar_processor]: Wall rejected: best_dot=0.45 < 0.7
```

**Cause:** Wall normals in maze frame don't match cardinals (because wrong theta)

**Fix:**
- Fix theta calculation first
- Then wall identification will work

### Issue 3: Normals Point Wrong Direction

**Symptoms:** Yellow arrows in RViz point away from robot

**Cause:** RANSAC normal direction is flipped

**Fix:** Line equation sign needs adjustment (uncommon, check RANSAC implementation)

## Testing Procedure

1. **Static Test:** Place robot in known position (e.g., corner at 0°)
   - Check if theta = 0° ± 5°
   - Check if walls identified correctly (left, bottom)

2. **Rotation Test:** Rotate robot 45°
   - Check if theta = 45° ± 5°
   - Check if same walls still identified correctly

3. **Position Test:** Move robot to different position in same cell
   - Check if theta stays consistent
   - Check if x, y position changes correctly

4. **Multi-cell Test:** Move to different cell
   - Check if only nearby walls (≤1m) are used
   - Check if position resets to new cell coordinates

## Next Steps

If the improved algorithm still fails:

1. **Collect failure case data:**
   - Copy all log output
   - Screenshot RViz showing misalignment
   - Note robot's actual position/orientation

2. **Analyze the logs:**
   - What are the detected wall angles?
   - What theta was calculated?
   - How many candidates did perpendicular constraint find?
   - What are the maze-frame normals?

3. **Potential improvements:**
   - Use wall length as additional constraint
   - Add temporal filtering (smooth theta over time)
   - Use odometry to bias orientation estimate
   - Implement particle filter for full pose estimation

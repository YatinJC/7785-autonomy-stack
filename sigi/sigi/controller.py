#!/usr/bin/env python3
"""
Cell center controller - drives robot to the center of its current cell.

Supports two modes:
1. Full localization (2+ perpendicular walls): Drive to cell center
2. Partial localization (1 wall or 2 parallel walls): Follow wall at center distance
"""

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from typing import Tuple, List, Dict
import json
import math
from enum import Enum
import numpy as np
from sigi_interfaces.srv import ReadSign

# Import geometry utilities
from . import geometry_utils as geom


class ControllerState(Enum):
    """States for the controller state machine."""
    WAITING = 0           # No position data yet
    ROTATING = 1          # Rotating to face center (full localization)
    CENTERING = 2         # Driving to center (full localization)
    DONE = 3              # At center (full localization)
    ALIGNING = 4          # Rotating to align parallel to wall(s) (partial localization)
    WALL_FOLLOWING = 5    # Driving forward parallel to wall(s) (partial localization)
    # Navigation States
    NAV_CENTERING = 6
    NAV_FACING_WALL = 7
    NAV_READING_SIGN = 8
    NAV_TURNING = 9
    NAV_DRIVING = 10
    NAV_SEARCHING_WALL = 11


class CellCenterController(Node):
    """ROS 2 node for controlling robot to center of cell."""
    
    # ========================================================================
    # CONSTANTS
    # ========================================================================
    
    # Velocity limits
    MAX_LINEAR_VEL = 0.15         # m/s
    MAX_ANGULAR_VEL = 0.5         # rad/s
    WALL_FOLLOW_SPEED = 0.1       # m/s - constant speed for wall following
    
    # Control gains
    KP_LINEAR = 0.5               # Proportional gain for linear velocity
    KP_ANGULAR = 1.0              # Proportional gain for angular velocity
    KP_LATERAL = 0.3              # Gain for lateral correction (wall following)
    
    # Tolerances
    POSITION_TOLERANCE = 0.05     # 5cm - close enough to center
    ANGLE_TOLERANCE = 0.1         # ~6 degrees
    
    # Localization thresholds
    MIN_CONFIDENCE = 0.3                      # Minimum confidence to act
    PARTIAL_LOCALIZATION_THRESHOLD = 0.5      # Confidence below this = partial
    
    # Safety
    POSITION_TIMEOUT = 1.0        # seconds - stop if no position update
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    def __init__(self):
        super().__init__('cell_center_controller')
        
        # Declare parameters
        self.declare_parameter('cell_size', 1.0)
        self.declare_parameter('kp_linear', self.KP_LINEAR)
        self.declare_parameter('kp_angular', self.KP_ANGULAR)
        self.declare_parameter('kp_lateral', self.KP_LATERAL)
        self.declare_parameter('max_linear_vel', self.MAX_LINEAR_VEL)
        self.declare_parameter('max_angular_vel', self.MAX_ANGULAR_VEL)
        self.declare_parameter('wall_follow_speed', self.WALL_FOLLOW_SPEED)
        self.declare_parameter('position_tolerance', self.POSITION_TOLERANCE)
        self.declare_parameter('angle_tolerance', self.ANGLE_TOLERANCE)
        self.declare_parameter('min_confidence', self.MIN_CONFIDENCE)
        self.declare_parameter('position_timeout', self.POSITION_TIMEOUT)
        
        # Get parameters
        self.cell_size = float(self.get_parameter('cell_size').value)
        self.kp_linear = float(self.get_parameter('kp_linear').value)
        self.kp_angular = float(self.get_parameter('kp_angular').value)
        self.kp_lateral = float(self.get_parameter('kp_lateral').value)
        self.max_linear_vel = float(self.get_parameter('max_linear_vel').value)
        self.max_angular_vel = float(self.get_parameter('max_angular_vel').value)
        self.wall_follow_speed = float(self.get_parameter('wall_follow_speed').value)
        self.position_tolerance = float(self.get_parameter('position_tolerance').value)
        self.angle_tolerance = float(self.get_parameter('angle_tolerance').value)
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.position_timeout = float(self.get_parameter('position_timeout').value)
        self.front_distance = float('inf')

        # State machines
        self.state = ControllerState.WAITING              # Low-level centering / wall-follow state
        self.nav_state = ControllerState.NAV_CENTERING    # High-level navigation state
        self.cell_pos = None
        self.walls_data = None  # Detailed wall information
        self.last_position_time = None
        self.target_theta = 0.0  # For wall following and driving
        self.centering_start_theta = None  # Saved orientation when entering NAV_CENTERING
        
        # Navigation State Variables
        self.sign_buffer: List[str] = []
        self.current_yaw = 0.0
        self.turn_start_yaw = 0.0
        self.turn_target_yaw = 0.0
        self.driving_direction = 0.0  # Direction we were driving before stopping (rad)
        self.facing_wall_reference_theta = None  # Saved orientation when entering NAV_FACING_WALL
        
        # Subscribers
        self.cell_position_sub = self.create_subscription(
            String,
            '/cell_position',
            self.cell_position_callback,
            10
        )
        
        self.detected_walls_sub = self.create_subscription(
            String,
            '/detected_walls',
            self.detected_walls_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Service client for on-demand sign reading
        self.read_sign_client = self.create_client(ReadSign, '/read_sign')

        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscriber
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            sensor_qos
        )
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Control loop timer (20 Hz)
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info('Cell center controller initialized')
    
    # ========================================================================
    # CALLBACKS
    # ========================================================================
    
    def cell_position_callback(self, msg: String):
        """Process incoming cell position data."""
        try:
            data = json.loads(msg.data)
            self.cell_pos = {
                'cell_x': float(data['cell_x']),
                'cell_y': float(data['cell_y']),
                'theta_rad': float(data['theta_rad']),
                'confidence': float(data['confidence']),
                'num_walls': int(data['num_walls'])
            }
            self.last_position_time = self.get_clock().now()
            
            # Log position updates
            self.get_logger().debug(
                f'Position: ({self.cell_pos["cell_x"]:.3f}, {self.cell_pos["cell_y"]:.3f}), '
                f'theta: {np.degrees(self.cell_pos["theta_rad"]):.1f}°, '
                f'conf: {self.cell_pos["confidence"]:.2f}, walls: {self.cell_pos["num_walls"]}'
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.get_logger().error(f'Failed to parse cell position: {e}')
    
    def detected_walls_callback(self, msg: String):
        """Process incoming detected walls data."""
        try:
            self.walls_data = json.loads(msg.data)
            self.get_logger().debug(
                f'Walls: {self.walls_data["num_walls"]} detected'
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.get_logger().error(f'Failed to parse detected walls: {e}')

    def odom_callback(self, msg: Odometry):
        """Process odometry for precise turning."""
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
    
    def control_loop(self):
        """Main control loop - runs at 20 Hz."""
        # Check if we have position data
        if self.cell_pos is None:
            self.publish_stop()
            if self.state != ControllerState.WAITING:
                self.get_logger().info('Waiting for position data...')
                self.state = ControllerState.WAITING
            return
        
        # Check if position data is fresh (watchdog)
        if self.last_position_time is not None:
            age = (self.get_clock().now() - self.last_position_time).nanoseconds / 1e9
            if age > self.position_timeout:
                self.get_logger().warn(f'Position data is stale ({age:.2f}s old), stopping')
                self.publish_stop()
                self.state = ControllerState.WAITING
                return
        
        # Check confidence threshold
        if self.cell_pos['confidence'] < self.min_confidence:
            self.get_logger().warn(
                f'Confidence too low ({self.cell_pos["confidence"]:.2f} < {self.min_confidence}), stopping'
            )
            self.publish_stop()
            return
        
        # Determine if we have full or partial localization
        has_full_localization = self._has_perpendicular_walls()
        
        if has_full_localization:
            self.get_logger().debug('Full localization: detected perpendicular walls')
        else:
            self.get_logger().debug('Partial localization: only parallel walls or single wall')
        
        # Run appropriate control logic
        if self.nav_state is not None:
            self.navigation_logic(has_full_localization)
        elif has_full_localization:
            self.full_localization_control()
        else:
            self.partial_localization_control()
            
    def navigation_logic(self, has_full_localization: bool):
        """High-level navigation state machine."""
        if self.cell_pos is None:
            self.publish_stop()
            return

        # 1. NAV_CENTERING: Use localization logic to center in the cell
        if self.nav_state == ControllerState.NAV_CENTERING:
            # Save orientation when first entering centering
            if self.centering_start_theta is None:
                self.centering_start_theta = self.cell_pos['theta_rad']
                self.get_logger().info(f'Entering NAV_CENTERING, saved theta: {np.degrees(self.centering_start_theta):.1f}°')
            # (Existing logic remains the same)
            if has_full_localization:
                if self.state not in [ControllerState.ROTATING, ControllerState.CENTERING, ControllerState.DONE]:
                    self.state = ControllerState.WAITING
                self.full_localization_control()
                if self.state == ControllerState.DONE and self._is_centered():
                    self.publish_stop()
                    self.get_logger().info(
                        f'Centered in cell. Transitioning to FACING_WALL '
                        f'(using saved theta: {np.degrees(self.centering_start_theta):.1f}°)'
                    )
                    self.nav_state = ControllerState.NAV_FACING_WALL
                    self.state = ControllerState.WAITING
                    self.sign_buffer = []
                    self.facing_wall_reference_theta = self.centering_start_theta
            else:
                if self.state not in [ControllerState.ALIGNING, ControllerState.WALL_FOLLOWING]:
                    self.state = ControllerState.WAITING
                self.partial_localization_control()
            return
        
        # 2. NAV_FACING_WALL: Rotate to face the wall
        if self.nav_state == ControllerState.NAV_FACING_WALL:
            # Use the saved orientation from when we entered this state
            if self.facing_wall_reference_theta is None:
                self.facing_wall_reference_theta = self.cell_pos['theta_rad']

            # Determine target cardinal based on reference orientation (only once)
            cardinals = [0.0, math.pi/2, math.pi, -math.pi/2]
            target_theta = min(cardinals, key=lambda x: abs(geom.normalize_angle(x - self.facing_wall_reference_theta)))

            # Calculate error using current orientation for control
            current_theta = self.cell_pos['theta_rad']
            error = geom.normalize_angle(target_theta - current_theta)
            
            if abs(error) < self.angle_tolerance:
                self.publish_stop()
                
                # CHECK: Is there actually a wall here?
                if self._check_wall_in_front():
                    self.get_logger().info(f'Facing wall ({np.degrees(target_theta):.0f}°). Reading sign...')
                    self.nav_state = ControllerState.NAV_READING_SIGN
                    self.sign_buffer = []
                else:
                    self.get_logger().info('Aligned to grid, but no wall. Starting 90° search turn...')
                    self.nav_state = ControllerState.NAV_SEARCHING_WALL
                    # Set target to 90 degrees LEFT of current orientation
                    self.turn_target_yaw = geom.normalize_angle(self.current_yaw + math.pi/2)
            else:
                # Standard rotation to face cardinal
                angular_vel = self.kp_angular * error
                angular_vel = self.clamp(angular_vel, -self.max_angular_vel, self.max_angular_vel)
                self.publish_velocity(0.0, angular_vel)
            return

        # 2.5 NAV_SEARCHING_WALL: Turn 90 deg -> Check -> Repeat
        if self.nav_state == ControllerState.NAV_SEARCHING_WALL:
            # Reuse the turn_target_yaw we set in the previous state
            error = geom.normalize_angle(self.turn_target_yaw - self.current_yaw)
            
            if abs(error) < self.angle_tolerance:
                self.publish_stop()
                
                # 1. The turn is done. NOW we check for the wall.
                if self._check_wall_in_front():
                    self.get_logger().info('Search turn complete. Wall found! Aligning...')
                    self.nav_state = ControllerState.NAV_FACING_WALL
                    self.facing_wall_reference_theta = self.cell_pos['theta_rad']
                else:
                    self.get_logger().info('Search turn complete. No wall. turning another 90°...')
                    # 2. No wall? Add another 90 degrees to the target and keep turning
                    self.turn_target_yaw = geom.normalize_angle(self.turn_target_yaw + math.pi/2)
            else:
                # Standard P-Control for turning (reuse existing gains)
                angular_vel = self.kp_angular * error
                angular_vel = self.clamp(angular_vel, -self.max_angular_vel, self.max_angular_vel)
                self.publish_velocity(0.0, angular_vel)
            return
        
        # 3. NAV_READING_SIGN: Collect votes using service
        if self.nav_state == ControllerState.NAV_READING_SIGN:
            # Call sign reading service if buffer not full
            if len(self.sign_buffer) < 10:
                # Check if service is available
                if not self.read_sign_client.service_is_ready():
                    self.get_logger().warn('Sign reading service not ready, waiting...')
                    return

                # Create and send request
                request = ReadSign.Request()
                future = self.read_sign_client.call_async(request)

                # Wait for result (blocking but quick)
                import rclpy
                rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

                if future.done():
                    try:
                        response = future.result()
                        if response.success:
                            self.sign_buffer.append(response.class_name)
                            self.get_logger().debug(
                                f'Sign reading {len(self.sign_buffer)}/10: {response.class_name} '
                                f'(conf: {response.confidence:.2f})'
                            )
                        else:
                            self.get_logger().warn('Sign reading service call failed')
                    except Exception as e:
                        self.get_logger().error(f'Service call exception: {e}')
                else:
                    self.get_logger().warn('Sign reading service call timed out')

            # Process buffer when full
            if len(self.sign_buffer) >= 10:
                from collections import Counter
                counts = Counter(self.sign_buffer)
                winner, _ = counts.most_common(1)[0]

                self.get_logger().info(f'Sign Vote Result: {dict(counts)} -> Winner: {winner}')

                if winner == 'goal':
                    self.get_logger().info('GOAL REACHED! Stopping.')
                    self.publish_stop()
                    self.nav_state = None
                    self.state = ControllerState.DONE
                elif winner in ['left', 'right', 'stop', 'do_not_enter']:
                    self.nav_state = ControllerState.NAV_TURNING
                    self.turn_start_yaw = self.current_yaw

                    # Determine turn target
                    if winner == 'left':
                        self.turn_target_yaw = geom.normalize_angle(self.current_yaw + math.pi/2)
                    elif winner == 'right':
                        self.turn_target_yaw = geom.normalize_angle(self.current_yaw - math.pi/2)
                    else:
                        self.turn_target_yaw = geom.normalize_angle(self.current_yaw + math.pi)

                    self.get_logger().info(f'Sign action: {winner}. Turning to {np.degrees(self.turn_target_yaw):.1f}°')
                else:
                    self.get_logger().warn('Saw EMPTY sign. Retrying...')
                    self.sign_buffer = []
            return
        
        # 4. NAV_TURNING: Execute turn
        if self.nav_state == ControllerState.NAV_TURNING:
            error = geom.normalize_angle(self.turn_target_yaw - self.current_yaw)
            
            if abs(error) < self.angle_tolerance:
                self.publish_stop()
                
                # LOOP CHECK: Is there a wall immediately in front?
                if self._check_wall_in_front():
                    self.get_logger().info('Turn complete. Wall detected immediately in front. Looping to READ SIGN.')
                    self.nav_state = ControllerState.NAV_FACING_WALL
                    self.facing_wall_reference_theta = self.cell_pos['theta_rad']
                else:
                    self.get_logger().info('Turn complete. Path clear. Driving...')
                    self.nav_state = ControllerState.NAV_DRIVING
                    # Snap driving direction to nearest cardinal for cleaner math
                    cardinals = [0.0, math.pi/2, math.pi, -math.pi/2]
                    self.driving_direction = min(cardinals, key=lambda x: abs(geom.normalize_angle(x - self.turn_target_yaw)))
            else:
                angular_vel = self.kp_angular * error
                angular_vel = self.clamp(angular_vel, -self.max_angular_vel, self.max_angular_vel)
                self.publish_velocity(0.0, angular_vel)
            return
        
        # 5. NAV_DRIVING: Drive straight until blocked
        if self.nav_state == ControllerState.NAV_DRIVING:
            wall_in_front = self._check_wall_in_front()
            
            if wall_in_front:
                self.publish_stop()
                self.get_logger().info('Wall detected in front. Centering...')
                self.nav_state = ControllerState.NAV_CENTERING
                self.state = ControllerState.WAITING
                self.centering_start_theta = None  # Reset so it gets saved fresh
            else:
                # SIMPLIFIED: Heading Control Only
                # Just keep the robot pointing in the driving_direction.
                # We ignore lateral (X/Y) error to prevent circling.
                
                self.target_theta = self.driving_direction
                current_theta = self.cell_pos['theta_rad']
                
                # Calculate simple heading error
                angular_error = geom.normalize_angle(self.target_theta - current_theta)
                
                # Calculate control commands
                # Note: We removed the lateral_cmd entirely
                linear_vel = self.wall_follow_speed
                angular_vel = self.kp_angular * angular_error
                
                # Clamp and publish
                angular_vel = self.clamp(angular_vel, -self.max_angular_vel, self.max_angular_vel)
                self.publish_velocity(linear_vel, 0)
            return
    
    # ========================================================================
    # FULL LOCALIZATION CONTROL
    # ========================================================================
    
    def full_localization_control(self):
        """Control logic when we have full localization (2+ perpendicular walls)."""
        cell_x = self.cell_pos['cell_x']
        cell_y = self.cell_pos['cell_y']
        theta = self.cell_pos['theta_rad']
        
        # Calculate error to center
        target_x = self.cell_size / 2
        target_y = self.cell_size / 2
        
        error_x = target_x - cell_x
        error_y = target_y - cell_y
        
        distance = math.sqrt(error_x**2 + error_y**2)
        angle_to_center = math.atan2(error_y, error_x)
        angular_error = self.normalize_angle(angle_to_center - theta)
        
        # State machine
        if self.state == ControllerState.WAITING:
            # First full localization, start rotating
            self.get_logger().info(f'Full localization acquired, distance to center: {distance:.3f}m')
            self.state = ControllerState.ROTATING
        
        if self.state == ControllerState.ROTATING:
            # Rotate to face center
            linear_vel = 0.0
            angular_vel = self.kp_angular * angular_error
            angular_vel = self.clamp(angular_vel, -self.max_angular_vel, self.max_angular_vel)
            
            self.publish_velocity(linear_vel, angular_vel)
            
            if abs(angular_error) < self.angle_tolerance:
                self.get_logger().info('Aligned with center, starting to drive')
                self.state = ControllerState.CENTERING
        
        elif self.state == ControllerState.CENTERING:
            # Drive to center
            linear_vel = self.kp_linear * distance
            linear_vel = self.clamp(linear_vel, 0.0, self.max_linear_vel)
            
            # Small angular correction
            angular_vel = self.kp_angular * angular_error
            angular_vel = self.clamp(angular_vel, -self.max_angular_vel/2, self.max_angular_vel/2)
            
            self.publish_velocity(linear_vel, angular_vel)
            
            if distance < self.position_tolerance:
                self.get_logger().info('Reached center!')
                self.state = ControllerState.DONE
        
        elif self.state == ControllerState.DONE:
            # At center, stop
            self.publish_stop()
            
            # Check if we drifted away
            if distance > self.position_tolerance * 2:
                self.get_logger().info(f'Drifted from center ({distance:.3f}m), re-centering')
                self.state = ControllerState.ROTATING
        
        elif self.state in [ControllerState.ALIGNING, ControllerState.WALL_FOLLOWING]:
            # Transitioning from partial to full localization
            self.get_logger().info('Upgraded to full localization, switching to centering mode')
            self.state = ControllerState.ROTATING
    
    # ========================================================================
    # PARTIAL LOCALIZATION CONTROL
    # ========================================================================
    
    def partial_localization_control(self):
        """Control logic when we have partial localization (1 wall or 2 parallel walls)."""
        cell_x = self.cell_pos['cell_x']
        cell_y = self.cell_pos['cell_y']
        theta = self.cell_pos['theta_rad']
        
        # Determine corridor orientation from wall slopes (much more reliable!)
        corridor_type, self.target_theta = self._determine_corridor_orientation()
        
        angular_error = self.normalize_angle(self.target_theta - theta)
        
        # State machine
        if self.state == ControllerState.WAITING:
            self.get_logger().info(
                f'Partial localization: {corridor_type}, '
                f'pos=({cell_x:.2f}, {cell_y:.2f}), '
                f'theta={np.degrees(theta):.1f}°, target={np.degrees(self.target_theta):.0f}°'
            )
            self.state = ControllerState.ALIGNING
        
        if self.state == ControllerState.ALIGNING:
            # Rotate to align parallel to wall
            linear_vel = 0.0
            angular_vel = self.kp_angular * angular_error
            angular_vel = self.clamp(angular_vel, -self.max_angular_vel, self.max_angular_vel)
            
            self.publish_velocity(linear_vel, angular_vel)
            
            if abs(angular_error) < self.angle_tolerance:
                self.get_logger().info(
                    f'Aligned with {corridor_type} '
                    f'(target: {np.degrees(self.target_theta):.0f}°), starting wall following'
                )
                self.state = ControllerState.WALL_FOLLOWING
        
        elif self.state == ControllerState.WALL_FOLLOWING:
            # Drive forward at constant speed, maintaining heading
            linear_vel = self.wall_follow_speed
            
            # Calculate lateral error (distance from center line)
            center = self.cell_size / 2
            lateral_error = 0.0
            
            if "horizontal" in corridor_type:
                lateral_error = center - cell_y
                if abs(self.target_theta) > math.pi/2:
                    lateral_error = -lateral_error
            else:
                lateral_error = center - cell_x
                if self.target_theta > 0:
                    lateral_error = -lateral_error
            
            heading_correction = self.kp_angular * angular_error
            lateral_correction = self.kp_lateral * lateral_error
            
            angular_vel = heading_correction + lateral_correction
            angular_vel = self.clamp(angular_vel, -self.max_angular_vel/4, self.max_angular_vel/4)
            
            self.publish_velocity(linear_vel, angular_vel)
            
            self.get_logger().debug(
                f'Wall following: lat_err={lateral_error:.3f}, ang_err={np.degrees(angular_error):.1f}°, '
                f'cmd={angular_vel:.3f}'
            )
        
        elif self.state in [ControllerState.ROTATING, ControllerState.CENTERING, ControllerState.DONE]:
            # Transitioning from full to partial localization
            self.get_logger().info('Downgraded to partial localization, switching to wall following')
            self.state = ControllerState.ALIGNING
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _has_perpendicular_walls(self) -> bool:
        """
        Check if we have perpendicular walls for full localization.
        
        Returns:
            True if we have 2+ perpendicular walls, False otherwise
        """
        if self.walls_data is None or self.walls_data['num_walls'] < 2:
            return False
        
        walls = self.walls_data['walls']
        
        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                angle1 = walls[i]['slope_rad']
                angle2 = walls[j]['slope_rad']
                if geom.are_angles_perpendicular(angle1, angle2, tolerance=0.2):
                    return True
        
        return False
    
    def _determine_corridor_orientation(self) -> Tuple[str, float]:
        """Determine corridor orientation from the detected walls."""
        if self.walls_data is None or self.walls_data['num_walls'] == 0:
            theta = self.cell_pos['theta_rad']
            if abs(theta) < math.pi/4 or abs(theta) > 3 * math.pi/4:
                return "horizontal (fallback)", 0.0
            return "vertical (fallback)", math.pi/2 if theta > 0 else -math.pi/2
        
        wall_slopes = [w['slope_rad'] for w in self.walls_data['walls']]
        avg_wall_slope = geom.normalize_angle(float(np.mean(wall_slopes)), 'pi')
        abs_slope = abs(avg_wall_slope)
        current_theta = self.cell_pos['theta_rad']
        
        if abs_slope < math.pi/4 or abs_slope > 3 * math.pi/4:
            # Wall is roughly horizontal (slope 0 or 180)
            # We should drive PARALLEL to it (East/West)
            candidates = [-math.pi, 0.0, math.pi]
            corridor_type = "horizontal corridor (horizontal walls)"
        else:
            # Wall is roughly vertical (slope 90 or -90)
            # We should drive PARALLEL to it (North/South)
            candidates = [math.pi/2, -math.pi/2]
            corridor_type = "vertical corridor (vertical walls)"
        
        target_theta = min(candidates, key=lambda x: abs(geom.normalize_angle(x - current_theta)))
        return corridor_type, target_theta
    
    def _is_centered(self) -> bool:
        """Check if robot is centered in the cell."""
        if not self.cell_pos:
            return False
        
        center = self.cell_size / 2
        tolerance = 0.1  # 10cm tolerance
        
        x_centered = abs(self.cell_pos['cell_x'] - center) < tolerance
        y_centered = abs(self.cell_pos['cell_y'] - center) < tolerance
        
        return x_centered and y_centered

    def _check_wall_in_front(self) -> bool:
        """
        Check if there is a wall directly in front using raw LiDAR data.
        Reliable and angle-invariant.
        """
        # Stop distance (0.7m from your previous code)
        STOP_DISTANCE = 0.7 
        
        # Check raw scan distance
        if self.front_distance < STOP_DISTANCE:
            return True
            
        return False

    def scan_callback(self, msg: LaserScan):
        """Process raw LiDAR data for fast obstacle detection."""
        ranges = np.array(msg.ranges)
        
        # Calculate angles for each range reading
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        
        # Define a cone in front of the robot (e.g., +/- 15 degrees)
        # Adjust FOV based on your robot's width and environment
        fov = np.radians(15) 
        
        # Create a mask for the front cone
        # We use simple angle wrapping logic here or assume -pi to pi
        # This logic works for standard -pi to pi scanners
        front_mask = np.abs(angles) < fov
        
        # Filter out invalid readings (inf, nan, zeros)
        valid_mask = np.isfinite(ranges) & (ranges > msg.range_min)
        
        # Combine masks
        mask = front_mask & valid_mask
        
        if np.any(mask):
            self.front_distance = np.min(ranges[mask])
        else:
            self.front_distance = float('inf')

    def publish_velocity(self, linear: float, angular: float):
        """Publish velocity command."""
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_vel_pub.publish(cmd)
    
    def publish_stop(self):
        """Publish zero velocity."""
        self.publish_velocity(0.0, 0.0)
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max."""
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


def main(args=None):
    """Main entry point."""
    import rclpy
    
    rclpy.init(args=args)
    node = CellCenterController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

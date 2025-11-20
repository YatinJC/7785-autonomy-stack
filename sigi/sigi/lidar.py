#!/usr/bin/env python3
"""
LiDAR processor node for wall detection and robot localization.

This node processes LiDAR scan data to:
1. Detect multiple walls using RANSAC
2. Validate wall geometry (perpendicular/parallel constraints)
3. Estimate robot position and orientation within a cell
4. Publish visualizations for RViz
"""

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from typing import Optional, Tuple, List, Dict
import json

# Import geometry utilities
from . import geometry_utils as geom


class lidar_processor(Node):
    """ROS 2 node for processing LiDAR scans and detecting walls."""
    
    # ========================================================================
    # CONSTANTS
    # ========================================================================
    
    # Wall detection thresholds
    WALL_ALIGNMENT_THRESHOLD = 0.6          # Minimum dot product for wall identification (lowered from 0.7)
    SINGLE_WALL_ALIGNMENT_THRESHOLD = 0.5   # Lower threshold for single wall case
    PARALLEL_ANGLE_THRESHOLD = 0.35        # ~20 degrees for parallel wall detection
    DUPLICATE_DISTANCE_THRESHOLD = 0.2      # 20cm threshold for duplicate detection
    
    # Orientation estimation thresholds
    THETA_AGREEMENT_TOLERANCE = 0.3         # ~17 degrees for perpendicular wall agreement
    CLUSTERING_THRESHOLD = 0.5              # ~30 degrees for angle clustering
    
    # Visualization constants
    MARKER_LINE_WIDTH = 0.02
    MARKER_NORMAL_SHAFT_DIAMETER = 0.02
    MARKER_NORMAL_HEAD_DIAMETER = 0.04
    MARKER_NORMAL_HEAD_LENGTH = 0.05
    MARKER_NORMAL_ARROW_LENGTH = 0.2        # 20cm arrow length
    WALL_COLORS = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    CELL_BOUNDARY_COLOR = (1.0, 1.0, 0.0)   # Yellow
    CELL_BOUNDARY_ALPHA = 0.8
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    def __init__(self):
        super().__init__('lidar_processor')
        
        # Configuration parameters
        self.declare_parameter('cell_size', 1.0)  # meters
        self.declare_parameter('max_walls', 3)
        self.declare_parameter('ransac_threshold', 0.03)  # meters
        self.declare_parameter('min_inliers', 15)
        self.declare_parameter('min_wall_length', 0.1)  # meters
        self.declare_parameter('angle_tolerance', 5.0)  # degrees
        self.declare_parameter('max_wall_distance', 1.0)  # meters
        
        self.cell_size = float(self.get_parameter('cell_size').value)
        self.max_walls = int(self.get_parameter('max_walls').value)
        self.ransac_threshold = float(self.get_parameter('ransac_threshold').value)
        self.min_inliers = int(self.get_parameter('min_inliers').value)
        self.min_wall_length = float(self.get_parameter('min_wall_length').value)
        self.angle_tolerance = np.radians(float(self.get_parameter('angle_tolerance').value))
        self.max_wall_distance = float(self.get_parameter('max_wall_distance').value)
        
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
        
        # Publishers - legacy (single line detection)
        self.line_pub = self.create_publisher(String, '/line_detection', 10)
        self.marker_pub = self.create_publisher(Marker, '/line_marker', 10)
        
        # Publishers - multi-wall detection and localization
        self.walls_pub = self.create_publisher(String, '/detected_walls', 10)
        self.cell_position_pub = self.create_publisher(String, '/cell_position', 10)
        self.walls_marker_pub = self.create_publisher(MarkerArray, '/wall_markers', 10)
        self.cell_boundary_pub = self.create_publisher(Marker, '/cell_boundary', 10)
    
    # ========================================================================
    # MAIN CALLBACK
    # ========================================================================
    
    def scan_callback(self, msg: LaserScan):
        """Process incoming LiDAR scan data."""
        # Convert polar to Cartesian coordinates
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        
        # Filter: valid ranges within 2 meters
        valid_mask = (ranges >= msg.range_min) & (ranges <= msg.range_max) & (ranges <= 2.0)
        x = ranges[valid_mask] * np.cos(angles[valid_mask])
        y = ranges[valid_mask] * np.sin(angles[valid_mask])
        
        if len(x) < 2:
            # self.get_logger().warn('Insufficient points for line detection')
            return
        
        # Multi-wall detection
        walls = self.detect_multiple_walls(x, y)
        
        if len(walls) == 0:
            # self.get_logger().warn('No walls detected')
            return
        
        # Validate wall geometry
        valid_walls, geometry_valid = self.validate_wall_geometry(walls)
        
        if not geometry_valid:
            # self.get_logger().warn('Wall geometry validation failed - angles don\'t match maze constraints')
            pass
        
        # Publish results
        self.publish_walls(valid_walls)
        self.publish_wall_markers(valid_walls, x, y)
        
        # Calculate and publish cell position
        if len(valid_walls) >= 1:
            cell_pos = self.calculate_cell_position(valid_walls)
            if cell_pos is not None:
                self.publish_cell_position(cell_pos)
                self.publish_cell_boundary(cell_pos)
            else:
                # self.get_logger().warn('Failed to calculate cell position')
                self.publish_cell_boundary(None)
        else:
            # self.get_logger().info('No valid walls detected for localization')
            self.publish_cell_boundary(None)
        
        # Legacy: Single line detection (for backward compatibility)
        result = self.ransac_line_fit(x, y)
        
        if result is None:
            return
        
        slope, inliers, line_params = result
        a, b, c = line_params
        distance = abs(c)
        
        # Determine which side the line is on
        inlier_mask = np.abs(a * x + b * y + c) < 0.05
        if np.sum(inlier_mask) > 0:
            mean_y = np.mean(y[inlier_mask])
            side = "left" if mean_y > 0 else "right"
        else:
            side = "unknown"
        
        # self.get_logger().info(
        #     f'Line detected - Slope: {slope:.4f} rad ({np.degrees(slope):.2f} deg), '
        #     f'Distance: {distance:.3f} m, Side: {side}, Inliers: {inliers}'
        # )
        
        # Publish legacy results
        msg_data = {
            'slope_rad': float(slope),
            'slope_deg': float(np.degrees(slope)),
            'distance_m': float(distance),
            'side': side,
            'inliers': int(inliers)
        }
        pub_msg = String()
        pub_msg.data = json.dumps(msg_data)
        self.line_pub.publish(pub_msg)
        
        self.publish_line_marker(x, y, a, b, c, inlier_mask)
    
    # ========================================================================
    # RANSAC AND WALL DETECTION
    # ========================================================================
    
    def ransac_line_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        iterations: int = 100,
        threshold: float = 0.05,
        min_inliers: int = 10
    ) -> Optional[Tuple[float, int, Tuple[float, float, float]]]:
        """
        RANSAC algorithm to find the best line through points.
        
        Args:
            x, y: Point coordinates
            iterations: Number of RANSAC iterations
            threshold: Distance threshold for inliers (meters)
            min_inliers: Minimum number of inliers to accept a line
        
        Returns:
            (slope, inliers_count, (a, b, c)) or None
            Line equation: ax + by + c = 0 (normalized)
        """
        points = np.column_stack([x, y])
        n_points = len(points)
        
        if n_points < 2:
            return None
        
        best_inliers_count = 0
        best_slope = None
        best_line_params = None
        
        for _ in range(iterations):
            # Randomly sample 2 points
            idx = np.random.choice(n_points, 2, replace=False)
            p1, p2 = points[idx]
            
            # Calculate line parameters: ax + by + c = 0
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Skip degenerate lines
            if np.hypot(dx, dy) < 1e-6:
                continue
            
            # Normal form: line perpendicular distance
            a = -dy
            b = dx
            c = -(a * p1[0] + b * p1[1])
            
            # Normalize
            norm = np.sqrt(a**2 + b**2)
            a, b, c = a/norm, b/norm, c/norm
            
            # Calculate distance from all points to the line
            distances = np.abs(a * x + b * y + c)
            
            # Count inliers
            inliers = np.sum(distances < threshold)
            
            # Update best model
            if inliers > best_inliers_count:
                best_inliers_count = inliers
                best_slope = np.arctan2(dy, dx)
                best_line_params = (a, b, c)
        
        if best_inliers_count >= min_inliers:
            return best_slope, best_inliers_count, best_line_params
        else:
            return None
    
    def detect_multiple_walls(self, x: np.ndarray, y: np.ndarray) -> List[Dict]:
        """
        Detect multiple walls using iterative RANSAC.
        
        Returns:
            List of wall dictionaries with keys:
            - 'slope': angle in radians
            - 'distance': perpendicular distance to wall
            - 'line_params': (a, b, c) for ax + by + c = 0
            - 'inliers': number of inlier points
            - 'inlier_mask': boolean mask of inlier points
            - 'length': length of the wall segment
            - 'inlier_points': (x, y) coordinates of inliers
        """
        walls = []
        remaining_x = x.copy()
        remaining_y = y.copy()
        
        for _ in range(self.max_walls):
            if len(remaining_x) < self.min_inliers:
                break
            
            # Run RANSAC on remaining points
            result = self.ransac_line_fit(
                remaining_x,
                remaining_y,
                iterations=100,
                threshold=self.ransac_threshold,
                min_inliers=self.min_inliers
            )
            
            if result is None:
                break
            
            slope, inliers_count, line_params = result
            a, b, c = line_params
            
            # Ensure normal points toward robot (origin)
            if c < 0:
                a, b, c = -a, -b, -c
            line_params = (a, b, c)
            
            distance = abs(c)
            
            # Skip walls too far away (belong to other cells)
            if distance > self.max_wall_distance:
                remaining_x, remaining_y = self._remove_inliers(
                    remaining_x, remaining_y, a, b, c
                )
                continue
            
            # Find inlier points
            distances_to_line = np.abs(a * remaining_x + b * remaining_y + c)
            inlier_mask = distances_to_line < self.ransac_threshold
            
            # Calculate wall length and validate
            if not self._is_wall_segment_valid(
                remaining_x[inlier_mask],
                remaining_y[inlier_mask],
                a, b, c
            ):
                remaining_x, remaining_y = remaining_x[~inlier_mask], remaining_y[~inlier_mask]
                continue
            
            # Calculate wall length
            wall_length = self._calculate_wall_length(
                remaining_x[inlier_mask],
                remaining_y[inlier_mask],
                a, b
            )
            
            # Check for duplicates
            if self._is_duplicate_wall(walls, slope, distance, line_params):
                remaining_x, remaining_y = remaining_x[~inlier_mask], remaining_y[~inlier_mask]
                continue
            
            # Add valid wall
            wall = {
                'slope': slope,
                'distance': distance,
                'line_params': line_params,
                'inliers': inliers_count,
                'inlier_mask': inlier_mask.copy(),
                'length': wall_length,
                'inlier_points': (
                    remaining_x[inlier_mask].copy(),
                    remaining_y[inlier_mask].copy()
                )
            }
            walls.append(wall)
            
            # Remove inliers from remaining points
            remaining_x, remaining_y = remaining_x[~inlier_mask], remaining_y[~inlier_mask]
        
        # Log detected walls
        for i, wall in enumerate(walls):
            # self.get_logger().info(
            #     f'Detected wall {i}: slope={np.degrees(wall["slope"]):.1f}°, '
            #     f'distance={wall["distance"]:.3f}m, length={wall["length"]:.3f}m, '
            #     f'normal=({wall["line_params"][0]:.3f}, {wall["line_params"][1]:.3f})'
            # )
            pass
        
        return walls
    
    def _remove_inliers(
        self,
        x: np.ndarray,
        y: np.ndarray,
        a: float,
        b: float,
        c: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove inlier points from remaining point cloud."""
        distances = np.abs(a * x + b * y + c)
        mask = distances >= self.ransac_threshold
        return x[mask], y[mask]
    
    def _is_wall_segment_valid(
        self,
        inlier_x: np.ndarray,
        inlier_y: np.ndarray,
        a: float,
        b: float,
        c: float
    ) -> bool:
        """Check if wall segment meets length and intersection criteria."""
        if len(inlier_x) < 2:
            return False
        
        # Calculate wall length
        line_dir_x = -b
        line_dir_y = a
        projections = inlier_x * line_dir_x + inlier_y * line_dir_y
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        wall_length = max_proj - min_proj
        
        # Clamp to cell size
        wall_length = min(wall_length, self.cell_size)
        
        # Check minimum length
        if wall_length < self.min_wall_length:
            # self.get_logger().info(
            #     f'Rejecting wall: length {wall_length:.3f}m < minimum {self.min_wall_length:.3f}m'
            # )
            return False
        
        # Check if normal from robot intersects the wall segment
        closest_point_x = -a * c
        closest_point_y = -b * c
        closest_point_proj = closest_point_x * line_dir_x + closest_point_y * line_dir_y
        
        if not (min_proj <= closest_point_proj <= max_proj):
            # self.get_logger().info(
            #     f'Rejecting wall: normal does not intersect segment '
            #     f'(proj={closest_point_proj:.3f} not in [{min_proj:.3f}, {max_proj:.3f}])'
            # )
            return False
        
        return True
    
    def _calculate_wall_length(
        self,
        inlier_x: np.ndarray,
        inlier_y: np.ndarray,
        a: float,
        b: float
    ) -> float:
        """Calculate wall segment length from inlier points."""
        line_dir_x = -b
        line_dir_y = a
        wall_length = geom.calculate_line_segment_length(
            inlier_x, inlier_y, line_dir_x, line_dir_y
        )
        return min(wall_length, self.cell_size)
    
    def _is_duplicate_wall(
        self,
        existing_walls: List[Dict],
        slope: float,
        distance: float,
        line_params: Tuple[float, float, float] # <--- Added argument
    ) -> bool:
        """Check if wall is a duplicate of an existing wall."""
        a1, b1, _ = line_params
        
        for existing_wall in existing_walls:
            angle_diff = geom.normalize_angle_diff(slope, existing_wall['slope'], 'pi')
            is_parallel = geom.are_angles_parallel(
                slope,
                existing_wall['slope'],
                self.PARALLEL_ANGLE_THRESHOLD
            )
            
            if is_parallel:
                dist_diff = abs(distance - existing_wall['distance'])
                if dist_diff < self.DUPLICATE_DISTANCE_THRESHOLD:
                    # NEW CHECK: Check if normals point in the same direction
                    # If dot product is positive, they are on the same side (Duplicate)
                    # If dot product is negative, they are on opposite sides (Distinct)
                    a2, b2, _ = existing_wall['line_params']
                    dot_product = a1 * a2 + b1 * b2
                    
                    if dot_product > 0:
                         # Same direction + Same distance = Duplicate
                        return True
        
        return False
    
    def validate_wall_geometry(self, walls: List[Dict]) -> Tuple[List[Dict], bool]:
        """
        Validate that walls satisfy maze geometry constraints.
        
        Walls should be either parallel (0° or 180°) or perpendicular (90°).
        
        Returns:
            (valid_walls, geometry_valid)
        """
        if len(walls) <= 1:
            return walls, True
        
        # Check all pairs of walls
        valid = True
        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                is_parallel = geom.are_angles_parallel(
                    walls[i]['slope'],
                    walls[j]['slope'],
                    self.angle_tolerance
                )
                is_perpendicular = geom.are_angles_perpendicular(
                    walls[i]['slope'],
                    walls[j]['slope'],
                    self.angle_tolerance
                )
                
                if not (is_parallel or is_perpendicular):
                    valid = False
                    angle_diff = geom.normalize_angle_diff(
                        walls[i]['slope'],
                        walls[j]['slope'],
                        'pi'
                    )
                    # self.get_logger().debug(
                    #     f'Wall pair {i},{j} has invalid angle: {np.degrees(angle_diff):.2f}°'
                    # )
        
        return walls, valid
    
    # ========================================================================
    # LOCALIZATION
    # ========================================================================
    
    def calculate_cell_position(self, walls: List[Dict]) -> Optional[Dict]:
        """
        Calculate robot position and orientation within cell.
        
        Uses wall angles to determine orientation and wall distances
        to determine position.
        
        Returns:
            Dictionary with 'x', 'y', 'theta', 'confidence', 'num_walls'
        """
        # Handle single wall case
        if len(walls) == 1:
            return self.calculate_position_single_wall(walls[0])
        elif len(walls) == 0:
            return None
        
        # Estimate robot orientation from wall angles
        robot_theta = self._estimate_robot_orientation(walls)
        
        # self.get_logger().info(
        #     f'Wall angles: {[np.degrees(w["slope"]) for w in walls]} deg, '
        #     f'Robot theta: {np.degrees(robot_theta):.1f}°'
        # )
        
        # Calculate position from wall distances
        cell_x, cell_y = self._calculate_position_from_walls(walls, robot_theta)
        
        # Calculate confidence
        confidence = min(1.0, len(walls) / 3.0)
        
        return {
            'x': cell_x,
            'y': cell_y,
            'theta': robot_theta,
            'confidence': confidence,
            'num_walls': len(walls)
        }
    
    def calculate_position_single_wall(self, wall: Dict) -> Optional[Dict]:
        """
        Calculate robot position using a single wall (lower confidence).
        
        Args:
            wall: Wall dictionary
        
        Returns:
            Position dictionary with confidence 0.4
        """
        wall_slope = wall['slope']
        wall_dist = wall['distance']
        a, b, c = wall['line_params']
        
        # self.get_logger().info(
        #     f'Single wall localization: slope={np.degrees(wall_slope):.1f}°, '
        #     f'distance={wall_dist:.3f}m'
        # )
        
        # Try aligning wall to each cardinal direction
        possible_configs = []
        
        for cardinal in [0, np.pi/2]:
            for offset in [0, np.pi]:
                maze_wall_angle = cardinal + offset
                robot_theta = geom.normalize_angle(maze_wall_angle - wall_slope, 'pi')
                
                possible_configs.append({
                    'robot_theta': robot_theta,
                    'maze_wall_angle': maze_wall_angle,
                    'cardinal': cardinal
                })
        
        # Pick configuration with robot_theta closest to 0
        best_config = min(possible_configs, key=lambda x: abs(x['robot_theta']))
        robot_theta = best_config['robot_theta']
        
        # self.get_logger().info(
        #     f'Selected configuration: robot_theta={np.degrees(robot_theta):.1f}°'
        # )
        
        # Transform wall normal to maze frame
        normal_maze_x, normal_maze_y = geom.rotate_vector(a, b, robot_theta)
        normal_maze_x, normal_maze_y = geom.normalize_vector(normal_maze_x, normal_maze_y)
        
        # Identify wall type
        match = geom.find_closest_cardinal(normal_maze_x, normal_maze_y)
        
        if match and match[3] > self.SINGLE_WALL_ALIGNMENT_THRESHOLD:
            cx, cy, wall_type, dot = match
            cell_x, cell_y = self._calculate_position_for_wall_type(
                wall_type, wall_dist
            )
            
            # self.get_logger().info(
            #     f'Single wall result: wall_type={wall_type}, '
            #     f'position=({cell_x:.3f}, {cell_y:.3f})m, '
            #     f'theta={np.degrees(robot_theta):.1f}°, dot={dot:.2f}'
            # )
            
            return {
                'x': cell_x,
                'y': cell_y,
                'theta': robot_theta,
                'confidence': 0.4,
                'num_walls': 1
            }
        else:
            # self.get_logger().warn('Single wall: could not identify wall type')
            return None
    
    def _estimate_robot_orientation(self, walls: List[Dict]) -> float:
        """Estimate robot orientation from wall angles."""
        # Find perpendicular wall pair
        ref_wall, perp_wall = self._find_perpendicular_wall_pair(walls)
        
        if perp_wall is not None:
            # Use perpendicular constraint
            theta = self._estimate_theta_from_perpendicular_walls(ref_wall, perp_wall)
            # self.get_logger().info(
            #     f'Using perpendicular constraint: theta={np.degrees(theta):.1f}°'
            # )
            return theta
        else:
            # Fallback: use wall slope alignment
            theta = self._estimate_theta_from_wall_slopes(walls)
            # self.get_logger().info(
            #     f'No perpendicular pair, using wall slope alignment: '
            #     f'theta={np.degrees(theta):.1f}°'
            # )
            return theta
    
    def _find_perpendicular_wall_pair(
        self,
        walls: List[Dict]
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Find the best perpendicular wall pair."""
        ref_wall = None
        perp_wall = None
        best_perp_score = float('inf')
        
        for i, wall_i in enumerate(walls):
            for wall_j in walls[i+1:]:
                angle_diff = geom.normalize_angle_diff(
                    wall_i['slope'],
                    wall_j['slope'],
                    'pi'
                )
                perp_error = abs(angle_diff - np.pi/2)
                
                if perp_error < self.angle_tolerance and perp_error < best_perp_score:
                    best_perp_score = perp_error
                    ref_wall = wall_i
                    perp_wall = wall_j
        
        if ref_wall is not None and perp_wall is not None:
            # self.get_logger().info(
            #     f'Found perpendicular pair: {np.degrees(ref_wall["slope"]):.1f}° and '
            #     f'{np.degrees(perp_wall["slope"]):.1f}°, error={np.degrees(best_perp_score):.1f}°'
            # )
            pass
        
        return ref_wall, perp_wall
    
    def _estimate_theta_from_perpendicular_walls(
        self,
        ref_wall: Dict,
        perp_wall: Dict
    ) -> float:
        """Estimate theta using perpendicular wall constraint."""
        angle_i = ref_wall['slope']
        angle_j = perp_wall['slope']
        
        possible_thetas = []
        
        for base_cardinal in [0, np.pi/2, np.pi, -np.pi/2]:
            # Try both orderings
            theta_from_i = geom.normalize_angle(base_cardinal - angle_i, 'pi')
            theta_from_j = geom.normalize_angle((base_cardinal + np.pi/2) - angle_j, 'pi')
            
            if abs(theta_from_i - theta_from_j) < self.THETA_AGREEMENT_TOLERANCE:
                possible_thetas.append((theta_from_i + theta_from_j) / 2)
            
            # Try other ordering
            theta_from_j_alt = geom.normalize_angle(base_cardinal - angle_j, 'pi')
            theta_from_i_alt = geom.normalize_angle((base_cardinal + np.pi/2) - angle_i, 'pi')
            
            if abs(theta_from_j_alt - theta_from_i_alt) < self.THETA_AGREEMENT_TOLERANCE:
                possible_thetas.append((theta_from_j_alt + theta_from_i_alt) / 2)
        
        if len(possible_thetas) > 0:
            clusters = geom.cluster_angles(possible_thetas, self.THETA_AGREEMENT_TOLERANCE)
            largest_cluster = geom.select_best_cluster(clusters, prefer_zero=True)
            return float(np.mean(largest_cluster))
        else:
            # self.get_logger().warn('Perpendicular constraint failed, using fallback')
            return 0.0
    
    def _estimate_theta_from_wall_slopes(self, walls: List[Dict]) -> float:
        """Estimate theta by aligning walls to cardinals (fallback method)."""
        possible_thetas = []
        
        for wall in walls:
            wall_slope = wall['slope']
            
            for cardinal_wall_angle in [0, np.pi/2]:
                theta1 = geom.normalize_angle(cardinal_wall_angle - wall_slope, 'pi')
                theta2 = geom.normalize_angle((cardinal_wall_angle + np.pi) - wall_slope, 'pi')
                possible_thetas.extend([theta1, theta2])
        
        # self.get_logger().info(
        #     f'Theta candidates: {[np.degrees(t) for t in possible_thetas]}'
        # )
        
        if len(possible_thetas) > 0:
            clusters = geom.cluster_angles(possible_thetas, self.CLUSTERING_THRESHOLD)
            largest_cluster = geom.select_best_cluster(clusters, prefer_zero=True)
            # self.get_logger().info(
            #     f'Found {len(clusters)} clusters, largest has {len(largest_cluster)} candidates'
            # )
            return float(np.mean(largest_cluster))
        else:
            return 0.0
    
    def _calculate_position_from_walls(
        self,
        walls: List[Dict],
        robot_theta: float
    ) -> Tuple[float, float]:
        """Calculate position from wall distances using geometric constraints."""
        x_estimates = []
        y_estimates = []
        
        for wall in walls:
            a, b, c = wall['line_params']
            wall_dist = wall['distance']
            
            # Transform wall normal to maze frame
            normal_maze_x, normal_maze_y = geom.rotate_vector(a, b, robot_theta)
            normal_maze_x, normal_maze_y = geom.normalize_vector(normal_maze_x, normal_maze_y)
            
            # Identify wall type
            match = geom.find_closest_cardinal(normal_maze_x, normal_maze_y)
            
            if match and match[3] > self.WALL_ALIGNMENT_THRESHOLD:
                cx, cy, wall_type, dot = match
                
                if wall_type == 'left':
                    x_estimates.append(wall_dist)
                elif wall_type == 'right':
                    x_estimates.append(self.cell_size - wall_dist)
                elif wall_type == 'bottom':
                    y_estimates.append(wall_dist)
                elif wall_type == 'top':
                    y_estimates.append(self.cell_size - wall_dist)
                
                # self.get_logger().info(
                #     f'Wall identified: {wall_type}, distance={wall_dist:.3f}m, '
                #     f'normal_maze=({normal_maze_x:.2f}, {normal_maze_y:.2f}), dot={dot:.2f}'
                # )
            else:
                # self.get_logger().warn(
                #     f'Wall rejected: best_dot={match[3] if match else 0:.2f} < {self.WALL_ALIGNMENT_THRESHOLD}'
                # )
                pass
        
        # Calculate final position
        cell_x = float(np.median(x_estimates)) if len(x_estimates) > 0 else self.cell_size / 2
        cell_y = float(np.median(y_estimates)) if len(y_estimates) > 0 else self.cell_size / 2
        
        # Clamp to cell bounds
        cell_x = max(0.0, min(self.cell_size, cell_x))
        cell_y = max(0.0, min(self.cell_size, cell_y))
        
        return cell_x, cell_y
    
    def _calculate_position_for_wall_type(
        self,
        wall_type: str,
        distance: float
    ) -> Tuple[float, float]:
        """Calculate position for a single wall type."""
        if wall_type == 'left':
            cell_x = distance
            cell_y = self.cell_size / 2
        elif wall_type == 'right':
            cell_x = self.cell_size - distance
            cell_y = self.cell_size / 2
        elif wall_type == 'bottom':
            cell_y = distance
            cell_x = self.cell_size / 2
        elif wall_type == 'top':
            cell_y = self.cell_size - distance
            cell_x = self.cell_size / 2
        else:
            cell_x = self.cell_size / 2
            cell_y = self.cell_size / 2
        
        # Clamp to bounds
        cell_x = max(0.0, min(self.cell_size, cell_x))
        cell_y = max(0.0, min(self.cell_size, cell_y))
        
        return cell_x, cell_y
    
    # ========================================================================
    # PUBLISHING AND VISUALIZATION
    # ========================================================================
    
    def publish_walls(self, walls: List[Dict]):
        """Publish detected walls information."""
        walls_data = {
            'num_walls': len(walls),
            'walls': []
        }
        
        for i, wall in enumerate(walls):
            # Extract normal vector (a, b) from line_params
            normal_a = float(wall['line_params'][0])
            normal_b = float(wall['line_params'][1])
            
            wall_info = {
                'id': i,
                'slope_rad': float(wall['slope']),
                'slope_deg': float(np.degrees(wall['slope'])),
                'distance_m': float(wall['distance']),
                'length_m': float(wall['length']),
                'inliers': int(wall['inliers']),
                'normal': [normal_a, normal_b]  # <--- ADD THIS LINE
            }
            walls_data['walls'].append(wall_info)
        
        msg = String()
        msg.data = json.dumps(walls_data)
        self.walls_pub.publish(msg)
    
    def publish_cell_position(self, cell_pos: Dict):
        """Publish cell position estimate."""
        msg = String()
        msg.data = json.dumps({
            'cell_x': cell_pos['x'],
            'cell_y': cell_pos['y'],
            'theta_rad': cell_pos['theta'],
            'theta_deg': float(np.degrees(cell_pos['theta'])),
            'confidence': cell_pos['confidence'],
            'num_walls': cell_pos['num_walls']
        })
        self.cell_position_pub.publish(msg)
        
        # self.get_logger().info(
        #     f'Cell position: ({cell_pos["x"]:.3f}, {cell_pos["y"]:.3f}) m, '
        #     f'theta: {np.degrees(cell_pos["theta"]):.1f}°, '
        #     f'confidence: {cell_pos["confidence"]:.2f}'
        # )
    
    def publish_wall_markers(self, walls: List[Dict], x: np.ndarray, y: np.ndarray):
        """Publish visualization markers for detected walls."""
        marker_array = MarkerArray()
        
        # Add wall line markers
        for i, wall in enumerate(walls):
            marker = self._create_wall_line_marker(wall, i)
            if marker:
                marker_array.markers.append(marker)
        
        # Add normal direction markers
        for i, wall in enumerate(walls):
            marker = self._create_wall_normal_marker(wall, i)
            if marker:
                marker_array.markers.append(marker)
        
        self.walls_marker_pub.publish(marker_array)
    
    def _create_wall_line_marker(self, wall: Dict, marker_id: int) -> Optional[Marker]:
        """Create a line marker for a wall."""
        a, b, c = wall['line_params']
        inlier_x, inlier_y = wall['inlier_points']
        
        if len(inlier_x) < 2:
            return None
        
        # Calculate line segment endpoints
        line_dir_x = -b
        line_dir_y = a
        
        projections = inlier_x * line_dir_x + inlier_y * line_dir_y
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        
        point_on_line_x = -a * c
        point_on_line_y = -b * c
        
        start_x = point_on_line_x + line_dir_x * min_proj
        start_y = point_on_line_y + line_dir_y * min_proj
        end_x = point_on_line_x + line_dir_x * max_proj
        end_y = point_on_line_y + line_dir_y * max_proj
        
        # Create marker
        marker = Marker()
        marker.header.frame_id = "base_scan"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "detected_walls"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        p1 = Point(x=float(start_x), y=float(start_y), z=0.0)
        p2 = Point(x=float(end_x), y=float(end_y), z=0.0)
        marker.points = [p1, p2]
        
        marker.scale.x = self.MARKER_LINE_WIDTH
        
        color = self.WALL_COLORS[marker_id % len(self.WALL_COLORS)]
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = 1.0
        
        marker.lifetime.sec = 1
        
        return marker
    
    def _create_wall_normal_marker(self, wall: Dict, marker_id: int) -> Optional[Marker]:
        """Create an arrow marker showing wall normal direction."""
        a, b, c = wall['line_params']
        
        # Normal arrow: from wall to robot
        start_x = -a * c
        start_y = -b * c
        end_x = start_x + a * self.MARKER_NORMAL_ARROW_LENGTH
        end_y = start_y + b * self.MARKER_NORMAL_ARROW_LENGTH
        
        marker = Marker()
        marker.header.frame_id = "base_scan"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "wall_normals"
        marker.id = 100 + marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        p1 = Point(x=float(start_x), y=float(start_y), z=0.0)
        p2 = Point(x=float(end_x), y=float(end_y), z=0.0)
        marker.points = [p1, p2]
        
        marker.scale.x = self.MARKER_NORMAL_SHAFT_DIAMETER
        marker.scale.y = self.MARKER_NORMAL_HEAD_DIAMETER
        marker.scale.z = self.MARKER_NORMAL_HEAD_LENGTH
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        marker.lifetime.sec = 1
        
        return marker
    
    def publish_cell_boundary(self, cell_pos: Optional[Dict]):
        """Publish visualization of the cell boundary in RViz."""
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "cell_boundary"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Get cell position or use default
        if cell_pos is not None:
            cell_x = cell_pos['x']
            cell_y = cell_pos['y']
            robot_theta = cell_pos['theta']
        else:
            cell_x = self.cell_size / 2
            cell_y = self.cell_size / 2
            robot_theta = 0.0
        
        # Create cell corners in maze frame
        corners_maze = [
            (0, 0),
            (self.cell_size, 0),
            (self.cell_size, self.cell_size),
            (0, self.cell_size),
            (0, 0)  # Close the square
        ]
        
        # Transform to robot frame
        for mx, my in corners_maze:
            # Translate
            tx = mx - cell_x
            ty = my - cell_y
            
            # Rotate (inverse rotation)
            rx, ry = geom.rotate_vector(tx, ty, -robot_theta)
            
            p = Point(x=float(rx), y=float(ry), z=0.0)
            marker.points.append(p)
        
        # Set marker properties
        marker.scale.x = self.MARKER_LINE_WIDTH
        marker.color.r = self.CELL_BOUNDARY_COLOR[0]
        marker.color.g = self.CELL_BOUNDARY_COLOR[1]
        marker.color.b = self.CELL_BOUNDARY_COLOR[2]
        marker.color.a = self.CELL_BOUNDARY_ALPHA
        
        marker.lifetime.sec = 1
        
        self.cell_boundary_pub.publish(marker)
    
    def publish_line_marker(
        self,
        x: np.ndarray,
        y: np.ndarray,
        a: float,
        b: float,
        c: float,
        inlier_mask: np.ndarray
    ):
        """Publish visualization marker for legacy single line detection."""
        x_inliers = x[inlier_mask]
        y_inliers = y[inlier_mask]
        
        if len(x_inliers) < 2:
            return
        
        # Calculate line segment extent
        line_dir_x = -b
        line_dir_y = a
        
        projections = x_inliers * line_dir_x + y_inliers * line_dir_y
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        
        point_on_line_x = -a * c
        point_on_line_y = -b * c
        
        start_x = point_on_line_x + line_dir_x * min_proj
        start_y = point_on_line_y + line_dir_y * min_proj
        end_x = point_on_line_x + line_dir_x * max_proj
        end_y = point_on_line_y + line_dir_y * max_proj
        
        # Create marker
        marker = Marker()
        marker.header.frame_id = "base_scan"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "detected_line"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        p1 = Point(x=float(start_x), y=float(start_y), z=0.0)
        p2 = Point(x=float(end_x), y=float(end_y), z=0.0)
        marker.points = [p1, p2]
        
        marker.scale.x = self.MARKER_LINE_WIDTH
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        marker.lifetime.sec = 1
        
        self.marker_pub.publish(marker)


def main(args=None):
    """Main entry point."""
    import rclpy
    
    rclpy.init(args=args)
    node = lidar_processor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
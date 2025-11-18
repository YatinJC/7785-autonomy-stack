#!/usr/bin/env python3
"""
Mapper Node for TurtleBot3 Autonomy Stack

This node builds an occupancy grid map from LiDAR observations and corrected odometry.
It accumulates obstacle points over time to create a persistent map of the environment.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose
import struct
import numpy as np
from collections import defaultdict


class MapperNode(Node):
    """
    ROS2 node for building an occupancy grid map from LiDAR observations.

    Subscribes to:
        /obstacle_points/PointCloud2 (PointCloud2): Global obstacle points
        /odom_corrected (Odometry): Robot pose for tracking explored areas

    Publishes:
        /map (OccupancyGrid): Occupancy grid map for RViz visualization
    """

    def __init__(self):
        super().__init__('mapper')

        # Map parameters
        self.resolution = 0.05  # meters per cell (5cm resolution)
        self.map_size = 200  # 200x200 cells = 10m x 10m map at 0.05m resolution
        self.origin_x = -5.0  # Map origin in world coordinates
        self.origin_y = -5.0

        # Occupancy grid storage
        # Dictionary: (grid_x, grid_y) -> observation_count
        self.obstacle_observations = defaultdict(int)
        self.free_observations = defaultdict(int)

        # Current robot position
        self.robot_x = 0.0
        self.robot_y = 0.0

        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to global obstacle points
        self.obstacle_sub = self.create_subscription(
            PointCloud2,
            '/obstacle_points/PointCloud2',
            self.obstacle_callback,
            sensor_qos
        )

        # Subscribe to corrected odometry for robot position
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom_corrected',
            self.odom_callback,
            10
        )

        # Publisher for occupancy grid map
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/map',
            10
        )

        # Timer to publish map at regular intervals (5Hz)
        self.map_timer = self.create_timer(0.2, self.publish_map)

        self.get_logger().info('Mapper Node started')
        self.get_logger().info(f'Map: {self.map_size}x{self.map_size} cells, resolution={self.resolution}m')

    def world_to_grid(self, x, y):
        """
        Convert world coordinates to grid cell indices.

        Args:
            x, y: World coordinates (meters)

        Returns:
            (grid_x, grid_y): Grid cell indices
        """
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        """
        Convert grid cell indices to world coordinates.

        Args:
            grid_x, grid_y: Grid cell indices

        Returns:
            (x, y): World coordinates (meters)
        """
        x = self.origin_x + (grid_x + 0.5) * self.resolution
        y = self.origin_y + (grid_y + 0.5) * self.resolution
        return (x, y)

    def is_valid_cell(self, grid_x, grid_y):
        """Check if grid cell is within map bounds."""
        return 0 <= grid_x < self.map_size and 0 <= grid_y < self.map_size

    def odom_callback(self, msg: Odometry):
        """
        Update current robot position from corrected odometry.

        Args:
            msg (Odometry): Corrected odometry message
        """
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def obstacle_callback(self, msg: PointCloud2):
        """
        Process obstacle points and update the map.

        For each obstacle point:
        1. Mark the obstacle cell as occupied
        2. Ray trace from robot to obstacle to mark free space

        Args:
            msg (PointCloud2): Global obstacle points
        """
        # Parse point cloud
        obstacle_points = self.parse_pointcloud2(msg)

        if len(obstacle_points) == 0:
            return

        # Get robot position in grid coordinates
        robot_grid = self.world_to_grid(self.robot_x, self.robot_y)

        for obs_x, obs_y in obstacle_points:
            # Convert obstacle to grid coordinates
            obs_grid = self.world_to_grid(obs_x, obs_y)

            if not self.is_valid_cell(obs_grid[0], obs_grid[1]):
                continue

            # Mark obstacle cell as occupied
            self.obstacle_observations[obs_grid] += 1

            # Ray trace from robot to obstacle to mark free space
            free_cells = self.bresenham_line(robot_grid[0], robot_grid[1],
                                            obs_grid[0], obs_grid[1])

            # Mark all cells along the ray (except the last one) as free
            for cell in free_cells[:-1]:
                if self.is_valid_cell(cell[0], cell[1]):
                    self.free_observations[cell] += 1

        self.get_logger().debug(
            f'Map updated: {len(self.obstacle_observations)} obstacle cells, '
            f'{len(self.free_observations)} free cells',
            throttle_duration_sec=1.0
        )

    def bresenham_line(self, x0, y0, x1, y1):
        """
        Bresenham's line algorithm for ray tracing through grid.

        Returns list of grid cells from (x0, y0) to (x1, y1).

        Args:
            x0, y0: Start cell coordinates
            x1, y1: End cell coordinates

        Returns:
            List of (x, y) tuples representing cells along the line
        """
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            cells.append((x, y))

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return cells

    def parse_pointcloud2(self, msg: PointCloud2):
        """
        Parse PointCloud2 message and extract (x, y) coordinates.

        Args:
            msg (PointCloud2): Input point cloud message

        Returns:
            list: List of (x, y) tuples
        """
        points = []
        point_step = msg.point_step

        for i in range(msg.width):
            offset = i * point_step
            x, y = struct.unpack_from('ff', msg.data, offset)
            points.append((x, y))

        return points

    def publish_map(self):
        """
        Publish the current occupancy grid map.

        Uses a probability model to convert observations to occupancy values:
        - Unknown: -1 (no observations)
        - Free: 0-49 (more free observations than obstacle observations)
        - Occupied: 50-100 (more obstacle observations than free observations)
        """
        # Create occupancy grid message
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'odom_corrected'

        # Set map metadata
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.map_size
        grid_msg.info.height = self.map_size
        grid_msg.info.origin.position.x = self.origin_x
        grid_msg.info.origin.position.y = self.origin_y
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

        # Initialize grid with unknown cells (-1)
        grid_data = np.full(self.map_size * self.map_size, -1, dtype=np.int8)

        # Fill in observed cells
        for grid_y in range(self.map_size):
            for grid_x in range(self.map_size):
                cell = (grid_x, grid_y)

                obstacle_obs = self.obstacle_observations.get(cell, 0)
                free_obs = self.free_observations.get(cell, 0)
                total_obs = obstacle_obs + free_obs

                if total_obs > 0:
                    # Calculate occupancy probability (0-100)
                    occupancy_prob = int((obstacle_obs / total_obs) * 100)

                    # Apply threshold: if more obstacle observations, mark as occupied
                    if obstacle_obs > free_obs:
                        # Occupied: 50-100
                        grid_data[grid_y * self.map_size + grid_x] = max(50, occupancy_prob)
                    else:
                        # Free: 0-49
                        grid_data[grid_y * self.map_size + grid_x] = min(49, occupancy_prob)

        grid_msg.data = grid_data.tolist()
        self.map_pub.publish(grid_msg)


def main(args=None):
    """
    Main entry point for the Mapper node.

    Initializes ROS2, creates the node instance, and spins until shutdown.

    Args:
        args: Command line arguments (optional)
    """
    rclpy.init(args=args)
    node = MapperNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

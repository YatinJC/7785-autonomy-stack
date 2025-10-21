#!/usr/bin/env python3
"""
Transform to Global Frame Node for TurtleBot3 Autonomy Stack

This node transforms LiDAR point cloud data from the robot's base_link frame
to the global (odom_corrected) frame using corrected odometry. The transformed
points are published for use by path planning nodes.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import numpy as np
import struct
import math


class TransformToGlobalNode(Node):
    """
    ROS2 node for transforming LiDAR point clouds to global coordinates.

    Subscribes to:
        /odom_corrected (Odometry): Corrected odometry data for robot pose
        /derez_lidar_base/PointCloud2 (PointCloud2): LiDAR points in base_link frame

    Publishes:
        /obstacle_points/PointCloud2 (PointCloud2): Transformed points in global frame

    The node applies a 2D transformation (rotation + translation) to convert each point
    from the robot's local frame to the global reference frame, enabling consistent
    obstacle mapping for path planning.
    """

    def __init__(self):
        """
        Initialize the TransformToGlobal node.

        Sets up subscriptions to odometry and LiDAR data, creates publisher for
        transformed points, and initializes state variables.
        """
        super().__init__('transform_to_global')

        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Store current robot pose from corrected odometry
        self.current_position = None
        self.current_yaw = None

        # Subscribe to corrected odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom_corrected',
            self.odom_callback,
            10
        )

        # Subscribe to LiDAR point cloud in base_link frame
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/derez_lidar_base/PointCloud2',
            self.lidar_callback,
            sensor_qos
        )

        # Publisher for transformed points in global frame
        self.global_points_pub = self.create_publisher(
            PointCloud2,
            '/obstacle_points/PointCloud2',
            sensor_qos
        )

        self.get_logger().info('Transform to Global node started')

    def odom_callback(self, msg: Odometry):
        """
        Callback for corrected odometry messages.

        Stores the current robot position and orientation for use in transforming
        LiDAR points to the global frame.

        Args:
            msg (Odometry): Corrected odometry message from /odom_corrected topic
        """
        self.current_position = msg.pose.pose.position

        # Extract yaw angle from quaternion orientation
        q = msg.pose.pose.orientation
        self.current_yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def lidar_callback(self, msg: PointCloud2):
        """
        Callback for LiDAR point cloud messages.

        Transforms each point from the robot's base_link frame to the global
        (odom_corrected) frame using the current robot pose, then publishes
        the transformed point cloud.

        Args:
            msg (PointCloud2): Point cloud in base_link frame from /derez_lidar_base/PointCloud2
        """
        # Wait until we have odometry data
        if self.current_position is None or self.current_yaw is None:
            self.get_logger().warn('Waiting for corrected odometry data...', throttle_duration_sec=5.0)
            return

        # Get robot's current pose in global frame
        robot_x = self.current_position.x
        robot_y = self.current_position.y
        robot_yaw = self.current_yaw

        # Parse incoming point cloud
        points_base = self.parse_pointcloud2(msg)

        if len(points_base) == 0:
            self.get_logger().warn('Received empty point cloud', throttle_duration_sec=5.0)
            return

        # Transform points from base_link to global frame
        transformed_points = []

        for x_base, y_base in points_base:
            # Apply 2D rotation and translation to transform from base_link to global frame
            x_global = robot_x + x_base * math.cos(robot_yaw) - y_base * math.sin(robot_yaw)
            y_global = robot_y + x_base * math.sin(robot_yaw) + y_base * math.cos(robot_yaw)

            # Round to nearest 0.1m for consistent grid representation
            # This ensures all obstacle points align to a 0.1m grid for path planning
            x_rounded = round(x_global, 1)
            y_rounded = round(y_global, 1)

            transformed_points.append((x_rounded, y_rounded))

        # Remove duplicate points
        transformed_points = set(transformed_points)

        # Create and publish the transformed point cloud
        global_cloud = self.create_pointcloud2(transformed_points, msg.header.stamp)
        self.global_points_pub.publish(global_cloud)

        self.get_logger().debug(
            f'Transformed {len(transformed_points)} points to global frame at pose '
            f'({robot_x:.2f}, {robot_y:.2f}, {robot_yaw:.2f})',
            throttle_duration_sec=1.0
        )

    def parse_pointcloud2(self, msg: PointCloud2):
        """
        Parse PointCloud2 message and extract (x, y) coordinates.

        Args:
            msg (PointCloud2): Input point cloud message

        Returns:
            list: List of (x, y) tuples representing point coordinates
        """
        points = []
        point_step = msg.point_step

        for i in range(msg.width):
            offset = i * point_step
            # Unpack x and y (two 32-bit floats)
            x, y = struct.unpack_from('ff', msg.data, offset)
            points.append((x, y))

        return points

    def create_pointcloud2(self, points, stamp):
        """
        Create a PointCloud2 message from a list of 2D points.

        Constructs a ROS2 PointCloud2 message containing (x, y) coordinates in the
        'odom_corrected' (global) frame.

        Args:
            points (list): List of (x, y) tuples representing points in global frame
            stamp (Time): Timestamp from the original LiDAR scan message

        Returns:
            PointCloud2: Formatted point cloud message ready for publishing
        """
        header = Header()
        header.stamp = stamp
        header.frame_id = 'odom_corrected'

        # Define the point cloud fields (x and y coordinates as 32-bit floats)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        ]

        # Pack each (x, y) point as binary data (two 32-bit floats)
        cloud_data = []
        for x, y in points:
            cloud_data.append(struct.pack('ff', x, y))

        # Assemble the PointCloud2 message
        pointcloud = PointCloud2()
        pointcloud.header = header
        pointcloud.height = 1  # Unorganized point cloud
        pointcloud.width = len(points)
        pointcloud.is_bigendian = False
        pointcloud.point_step = 8  # 2 floats * 4 bytes each = 8 bytes per point
        pointcloud.row_step = pointcloud.point_step * len(points)
        pointcloud.fields = fields
        pointcloud.is_dense = True  # No invalid points in the cloud
        pointcloud.data = b''.join(cloud_data)

        return pointcloud


def main(args=None):
    """
    Main entry point for the TransformToGlobal node.

    Initializes ROS2, creates the node instance, and spins until shutdown.
    Handles keyboard interrupts gracefully and ensures proper cleanup.

    Args:
        args: Command line arguments (optional)
    """
    rclpy.init(args=args)
    node = TransformToGlobalNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

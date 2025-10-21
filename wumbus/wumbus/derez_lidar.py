#!/usr/bin/env python3
"""
LiDAR Processing Node for TurtleBot3 Autonomy Stack

This node processes laser scan data by transforming it from the scanner's frame
to the robot's base_link frame and publishing it as a PointCloud2 message.
It filters out invalid measurements and applies range limits.
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener
import math
import struct
import numpy as np


class DerezLidar(Node):
    """
    ROS2 node for processing LiDAR scan data.

    Subscribes to:
        /scan (LaserScan): Raw laser scan data from the LiDAR sensor
        /odom_corrected (Odometry): Corrected odometry data for the robot

    Publishes:
        /derez_lidar_base/PointCloud2 (PointCloud2): Transformed point cloud in base_link frame

    The node transforms laser scan measurements from the base_scan frame to the base_link frame,
    applies range filtering (max 2.0m), and publishes the result as a point cloud.
    """
    def __init__(self):
        """
        Initialize the DerezLidar node.

        Sets up TF2 listeners for coordinate transformations, initializes odometry storage,
        creates subscriptions to scan and odometry topics, and sets up the point cloud publisher.
        """
        super().__init__('derez_lidar')
        # Debug flag to log transformed points once at startup
        self.flag = True

        # TF2 setup for coordinate frame transformations (base_scan to base_link)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Store the latest corrected odometry data (for sync checking)
        self.current_position = None
        self.current_yaw = None

        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to corrected odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom_corrected',
            self.odom_callback,
            10
        )

        # Subscribe to laser scan
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            sensor_qos
        )

        # Publisher for transformed points
        self.full_pointcloud_pub = self.create_publisher(
            PointCloud2,
            '/derez_lidar_base/PointCloud2',
            sensor_qos
        )

        self.derez_pointcloud_pub = self.create_publisher(
            PointCloud2,
            '/derez_lidar_base/PointCloud2',
            sensor_qos
        )

        self.get_logger().info('Derez LiDAR node started')

    def odom_callback(self, msg: Odometry):
        """
        Callback for corrected odometry messages.

        Stores the latest position and yaw for synchronization checking.
        The node waits for odometry data before processing scan data to ensure
        the transform is available.

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

    def scan_callback(self, msg: LaserScan):
        """
        Callback for laser scan messages.

        Processes incoming laser scan data by:
        1. Converting polar coordinates (range, angle) to Cartesian coordinates (x, y)
        2. Transforming points from base_scan frame to base_link frame using TF2
        3. Filtering out invalid measurements (NaN, inf, out of range)
        4. Applying a maximum range limit of 2.0 meters
        5. Publishing the transformed points as a PointCloud2 message

        Args:
            msg (LaserScan): Laser scan message from the /scan topic
        """

        # Look up the static transform from base_link to base_scan
        # This accounts for the LiDAR sensor's mounting position on the robot
        try:
            # Request the transform with a 100ms timeout
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                'base_scan',
                Time(),
                timeout=Duration(seconds=0, nanoseconds=100000000)
            )
            # Extract translation and rotation from the transform
            scan_offset_x = transform.transform.translation.x
            scan_offset_y = transform.transform.translation.y
            q = transform.transform.rotation
            # Convert quaternion to yaw angle
            scan_offset_yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
        except Exception as e:
            # If transform is unavailable, assume the LiDAR is mounted at base_link origin
            self.get_logger().debug(f'Using base_link frame (no base_scan transform): {e}')
            scan_offset_x = 0.0
            scan_offset_y = 0.0
            scan_offset_yaw = 0.0

        # Process each laser scan measurement
        transformed_points = []

        for i, range_val in enumerate(msg.ranges):
            # Skip invalid range measurements
            if math.isnan(range_val) or math.isinf(range_val):
                continue
            # Apply range filtering: enforce minimum range and 2.0m maximum
            if range_val < msg.range_min or range_val > 2.0:
                continue

            # Calculate the angle for this specific ray
            angle = msg.angle_min + i * msg.angle_increment

            # Convert from polar (range, angle) to Cartesian (x, y) in base_scan frame
            x_scan = range_val * math.cos(angle)
            y_scan = range_val * math.sin(angle)

            # Apply 2D rotation and translation to transform from base_scan to base_link frame
            x_base = scan_offset_x + x_scan * math.cos(scan_offset_yaw) - y_scan * math.sin(scan_offset_yaw)
            y_base = scan_offset_y + x_scan * math.sin(scan_offset_yaw) + y_scan * math.cos(scan_offset_yaw)

            x_rounded = round(x_base, 1)
            y_rounded = round(y_base, 1)

            transformed_points.append((x_rounded, y_rounded))


        # Store transformed points as a set (removes duplicates)
        self.transformed_points = set(transformed_points)

        # Log transformed points once at startup for debugging
        # if self.flag == True:
        #     self.get_logger().info(f'Transformed points: {self.transformed_points}')
        #     self.flag = False

        # Create and publish the PointCloud2 message with transformed points
        pointcloud_msg = self.create_pointcloud2(self.transformed_points, msg.header.stamp)
        self.full_pointcloud_pub.publish(pointcloud_msg)

        

    def create_pointcloud2(self, points, stamp):
        """
        Create a PointCloud2 message from a set of 2D points.

        Constructs a ROS2 PointCloud2 message containing (x, y) coordinates in the
        'base_link' frame. Each point is stored as two 32-bit floats.

        Args:
            points (set): Set of (x, y) tuples representing points in base_link frame
            stamp (Time): Timestamp from the original laser scan message

        Returns:
            PointCloud2: Formatted point cloud message ready for publishing
        """
        header = Header()
        header.stamp = stamp
        header.frame_id = 'base_link'

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
    Main entry point for the DerezLidar node.

    Initializes ROS2, creates the node instance, and spins until shutdown.
    Handles keyboard interrupts gracefully and ensures proper cleanup.

    Args:
        args: Command line arguments (optional)
    """
    rclpy.init(args=args)
    node = DerezLidar()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener
# from tf2_ros import TransformException
from collections import Counter
import math
import struct
import numpy as np


class DerezLidar(Node):
    def __init__(self):
        super().__init__('derez_lidar')
        self.flag = True

        # TF2 setup for base_scan to base_link transform (static transform)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Store current corrected odometry
        self.current_odom = None
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
        self.pointcloud_pub = self.create_publisher(
            PointCloud2,
            '/obstacle_points/PointCloud2',
            sensor_qos
        )

        self.get_logger().info('Derez LiDAR node started')

    def odom_callback(self, msg: Odometry):
        """
        Store the latest corrected odometry data
        """
        self.current_odom = msg
        self.current_position = msg.pose.pose.position

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        self.current_yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def scan_callback(self, msg: LaserScan):
        """
        Process laser scan data: convert to x,y coordinates, transform to global frame
        using corrected odometry, and round to nearest 0.1
        """
        # Check if we have corrected odometry data
        if self.current_position is None or self.current_yaw is None:
            self.get_logger().warn('Waiting for corrected odometry data...', throttle_duration_sec=5.0)
            return

        # Get robot's position and orientation in global frame from corrected odom
        robot_x = self.current_position.x
        robot_y = self.current_position.y
        robot_yaw = self.current_yaw

        # Try to get transform from base_link to base_scan (usually static)
        # This accounts for the LiDAR's position on the robot
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                'base_scan',
                Time(),
                timeout=Duration(seconds=0, nanoseconds=100000000)
            )
            # Extract offset from base_link to base_scan
            scan_offset_x = transform.transform.translation.x
            scan_offset_y = transform.transform.translation.y
            q = transform.transform.rotation
            scan_offset_yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
        except Exception as e:
            # If transform not available, assume scan is at base_link
            self.get_logger().debug(f'Using base_link frame (no base_scan transform): {e}')
            scan_offset_x = 0.0
            scan_offset_y = 0.0
            scan_offset_yaw = 0.0

        # Process each laser scan point
        transformed_points = []

        for i, range_val in enumerate(msg.ranges):
            # Filter out invalid points
            if math.isnan(range_val) or math.isinf(range_val):
                continue
            if range_val < msg.range_min or range_val > 2.0:
                continue

            # Calculate angle for this measurement
            angle = msg.angle_min + i * msg.angle_increment

            # Convert polar to Cartesian in base_scan frame
            x_scan = range_val * math.cos(angle)
            y_scan = range_val * math.sin(angle)

            # Transform from base_scan to base_link
            x_base = scan_offset_x + x_scan * math.cos(scan_offset_yaw) - y_scan * math.sin(scan_offset_yaw)
            y_base = scan_offset_y + x_scan * math.sin(scan_offset_yaw) + y_scan * math.cos(scan_offset_yaw)

            # Transform from base_link to global frame using corrected odometry
            x_global = robot_x + x_base * math.cos(robot_yaw) - y_base * math.sin(robot_yaw)
            y_global = robot_y + x_base * math.sin(robot_yaw) + y_base * math.cos(robot_yaw)

            # Round to nearest 0.1
            x_rounded = round(x_global, 1)
            y_rounded = round(y_global, 1)

            transformed_points.append((x_rounded, y_rounded))

        # Store the transformed points (for later use)
        self.transformed_points = set(transformed_points)
        
        
        if self.flag == True:
            self.get_logger().info(f'Transformed points: {self.transformed_points}')
            self.flag = False

        # Create and publish PointCloud2 message
        pointcloud_msg = self.create_pointcloud2(self.transformed_points, msg.header.stamp)
        self.pointcloud_pub.publish(pointcloud_msg)

        

    def create_pointcloud2(self, points, stamp):
        """
        Create a PointCloud2 message from a list of (x, y) tuples in global frame
        """
        header = Header()
        header.stamp = stamp
        header.frame_id = 'odom_corrected'

        # Define fields for x, y, z (z will be 0)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        ]

        # Pack point data
        cloud_data = []
        for x, y in points:
            cloud_data.append(struct.pack('ff', x, y))

        # Create PointCloud2 message
        pointcloud = PointCloud2()
        pointcloud.header = header
        pointcloud.height = 1
        pointcloud.width = len(points)
        pointcloud.is_bigendian = False
        pointcloud.point_step = 8  # 3 floats * 4 bytes each
        pointcloud.row_step = pointcloud.point_step * len(points)
        pointcloud.fields = fields
        pointcloud.is_dense = True
        pointcloud.data = b''.join(cloud_data)

        return pointcloud


def main(args=None):
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

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener
# from tf2_ros import TransformException
from collections import Counter
import math
import struct


class DerezLidar(Node):
    def __init__(self):
        super().__init__('derez_lidar')
        self.flag = True
        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
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

    def scan_callback(self, msg: LaserScan):
        """
        Process laser scan data: convert to x,y coordinates, transform to /odom frame,
        and round to nearest 0.1
        """
        try:
            # Look up transform from base_scan to odom at the time of the scan
            # scan_time = Time.from_msg(msg.header.stamp)
            transform = self.tf_buffer.lookup_transform(
                'odom',
                'base_scan',
                Time(),
                timeout=Duration(seconds=0, nanoseconds=100000000)
            )
        except Exception as e:
            self.get_logger().warn(f'Transform lookup failed: {e}', throttle_duration_sec=5.0)
            return

        # Extract transform components
        trans = transform.transform.translation
        rot = transform.transform.rotation

        # Convert quaternion to yaw angle
        yaw = math.atan2(
            2.0 * (rot.w * rot.z + rot.x * rot.y),
            1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z)
        )

        # Process each laser scan point
        transformed_points = []

        for i, range_val in enumerate(msg.ranges):
            # Filter out invalid points
            if math.isnan(range_val) or math.isinf(range_val):
                continue
            if range_val < msg.range_min or range_val > msg.range_max:
                continue

            # Calculate angle for this measurement
            angle = msg.angle_min + i * msg.angle_increment

            # Convert polar to Cartesian in base_scan frame
            x_base = range_val * math.cos(angle)
            y_base = range_val * math.sin(angle)

            # Transform to odom frame
            x_odom = trans.x + x_base * math.cos(yaw) - y_base * math.sin(yaw)
            y_odom = trans.y + x_base * math.sin(yaw) + y_base * math.cos(yaw)

            # Round to nearest 0.1
            x_rounded = round(x_odom, 1)
            y_rounded = round(y_odom, 1)

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
        Create a PointCloud2 message from a list of (x, y) tuples
        """
        header = Header()
        header.stamp = stamp
        header.frame_id = 'odom'

        # Define fields for x, y, z (z will be 0)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        ]

        # Pack point data
        cloud_data = []
        for x, y in points:
            cloud_data.append(struct.pack('fff', x, y, 0.0))

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

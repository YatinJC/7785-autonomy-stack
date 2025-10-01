#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from custom_msgs.msg import Track

class get_object_range(Node):
    def __init__(self):
        super().__init__("get_object_range")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        # Subscriptions
        self.create_subscription(Float32, "/x_value/Float32", self.centroid_callback, 10)
        self.create_subscription(Float32, "/obj_width/Float32", self.width_callback, 10)
        self.create_subscription(LaserScan, "/scan", self.scan_callback, qos)

        # Publisher
        self.pub = self.create_publisher(Track, '/track/Custom_Track', 10)

        # Initialisation variables
        self.scan = None
        self.theta = None       # centroid angle (radians)
        self.width = None       # wrapper width (radians)
        self.d = None
        # System Definations
        self.camera_fov = np.deg2rad(60)


    # -----------------------------------------------------
    # 1. Convert normalized centroid [-1,1] → radians
    # -----------------------------------------------------
    def update_centroid(self, norm_val):
        norm_val = - max(-1.0, min(1.0, norm_val))

        self.theta = norm_val * (self.camera_fov / 2.0)
        if self.theta < 0:
            self.theta += 2*np.pi

        
    def centroid_callback(self, msg: Float32):

        self.update_centroid(msg.data)

    # -----------------------------------------------------
    # 2. Convert normalized width [-1,1] → radians
    # -----------------------------------------------------
    def update_width(self, width_val):

        width_val = max(0.0, min(1.0, width_val))
        self.width = width_val * self.camera_fov / 2.0

    def width_callback(self, msg: Float32):

        self.update_width(msg.data)

    # -----------------------------------------------------
    # 3. Process scan → filter window [θ-φ/2, θ+φ/2], average range
    # -----------------------------------------------------
    def process_scan(self):
        if self.scan is None or self.theta is None or self.width is None:
            return

        theta_min = self.theta - self.width
        if theta_min < 0:
            theta_min += 2*np.pi
        theta_max = self.theta + self.width
        if theta_max >= 2*np.pi:
            theta_max -= 2*np.pi

        self.get_logger().info(f'theta_min: {theta_min:.4f}, theta_max: {theta_max:.4f}')

        num_points = len(self.scan.ranges)
        all_angles = self.scan.angle_min + np.arange(num_points) * self.scan.angle_increment

        ranges = np.array(self.scan.ranges)


        valid_mask = (ranges > self.scan.range_min) & (ranges < self.scan.range_max)
        ranges = ranges[valid_mask]
        angles = all_angles[valid_mask]

        window_mask = (angles >= theta_min) & (angles <= theta_max)
        filtered_ranges = ranges[window_mask]

        if len(filtered_ranges) == 0:
            self.get_logger().warn("No valid LiDAR points in the angular window")
            return

        self.d = float(min(filtered_ranges))


    def scan_callback(self, msg: LaserScan):
        self.scan = msg
        self.process_scan()

        if self.theta is not None and self.d is not None:
            track_msg = Track()
            track_msg.theta = float(self.theta)
            track_msg.d = float(self.d)

            self.pub.publish(track_msg)


def main(args=None):
    rclpy.init(args=args)
    node = get_object_range()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

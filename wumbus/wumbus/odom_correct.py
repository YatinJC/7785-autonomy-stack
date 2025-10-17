#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
import numpy as np


class OdometryCorrectNode(Node):
    """
    ROS 2 node that subscribes to raw odometry data, corrects it by
    zeroing out the initial position and orientation, and publishes
    the corrected odometry.
    """

    def __init__(self):
        super().__init__('odom_correct')

        # Subscribe to raw odometry
        self.subscription_odom = self.create_subscription(
            Odometry,
            '/odom',
            self.update_Odometry,
            10)

        # Publish corrected odometry
        self.odom_pub = self.create_publisher(Odometry, '/odom_corrected', 10)

        # Initialize position tracking variables
        self.Init = True
        self.Init_ang = 0.0
        self.globalAng = 0.0
        self.Init_pos = Point()
        self.globalPos = Point()

        self.get_logger().info('Odometry Correction Node started')

    def update_Odometry(self, Odom):
        """
        Callback function that processes incoming odometry data,
        corrects it relative to the initial position/orientation,
        and publishes the corrected odometry.
        """
        position = Odom.pose.pose.position

        # Orientation uses the quaternion parametrization.
        # To get the angular position along the z-axis, the following equation is required.
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

        if self.Init:
            # The initial data is stored to be subtracted from all the other values
            # as we want to start at position (0,0) and orientation 0
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],
                             [-np.sin(self.Init_ang), np.cos(self.Init_ang)]])
            self.Init_pos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Init_pos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Init_pos.z = position.z
            self.get_logger().info(f'Initial odometry set: pos=({self.Init_pos.x:.3f}, {self.Init_pos.y:.3f}), ang={self.Init_ang:.3f}')

        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],
                         [-np.sin(self.Init_ang), np.cos(self.Init_ang)]])

        # We subtract the initial values
        self.globalPos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos.x
        self.globalPos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos.y
        self.globalAng = orientation - self.Init_ang

        # Create and publish corrected odometry message
        corrected_odom = Odometry()
        corrected_odom.header = Odom.header
        corrected_odom.header.frame_id = 'odom_corrected'
        corrected_odom.child_frame_id = Odom.child_frame_id

        # Set corrected position
        corrected_odom.pose.pose.position.x = self.globalPos.x
        corrected_odom.pose.pose.position.y = self.globalPos.y
        corrected_odom.pose.pose.position.z = position.z

        # Convert corrected angle back to quaternion
        corrected_odom.pose.pose.orientation.x = 0.0
        corrected_odom.pose.pose.orientation.y = 0.0
        corrected_odom.pose.pose.orientation.z = np.sin(self.globalAng / 2.0)
        corrected_odom.pose.pose.orientation.w = np.cos(self.globalAng / 2.0)

        # Copy velocity data (unchanged)
        corrected_odom.twist = Odom.twist

        # Copy covariance (unchanged)
        corrected_odom.pose.covariance = Odom.pose.covariance

        # Publish corrected odometry
        self.odom_pub.publish(corrected_odom)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryCorrectNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

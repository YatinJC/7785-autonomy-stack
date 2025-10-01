#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math
from custom_msgs.msg import Track

class chase_object(Node):
    def __init__(self):
        super().__init__('chase_object')

        # Default k value
        self.k = 2.0

        # Initialising

        self.theta = 0.0
        self.d = 1.0
        self.l = 0.0

        # QoS profile for distributed communication
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.create_subscription(Track, '/track/Custom_Track', self.track_callback, 10)

        # Publisher
        self.publisher = self.create_publisher(Twist, '/cmd_vel', qos)

    def track_callback(self, msg):
        self.theta = msg.theta
        self.d = msg.d

        self.get_logger().info(f'Received Track - theta: {self.theta:.4f}, d: {self.d:.4f}')

        self.controller()
        self.publish_twist()


    def controller(self):
        R = 0.033
        b = 0.16

        phi_r = 6*R/(1+(math.sin(self.theta)*b/self.d))
        phi_l = 6*R/(1-(math.sin(self.theta)*b/self.d))

        self.l = self.k*(self.d - 0.5)
        if 30 > self.d > 20:
            self.l = 0 
        self.lmax = min(phi_r,phi_l)
        if self.l > self.lmax:
            self.l = self.lmax

    def publish_twist(self):
        twist = Twist()

        twist.linear.x = self.l
        twist.angular.z = 2*abs(self.l)*math.sin(self.theta)/self.d

        self.get_logger().info(f'Publishing Twist - linear.x: {twist.linear.x:.4f}, angular.z: {twist.angular.z:.4f}')
        self.publisher.publish(twist)

    def stop_robot(self):
        """Publish a zero velocity command to stop the robot."""
        twist = Twist()
        # All fields default to 0.0
        self.get_logger().info('Stopping robot - publishing zero velocity')
        self.publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = chase_object()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("ChaseObject stopped cleanly (Ctrl+C).")
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
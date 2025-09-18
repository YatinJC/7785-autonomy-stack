#!/usr/bin/env python3
# Yatin Chandar and Arun Showry Busani

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class RotateRobot(Node):
    def __init__(self):
        super().__init__('rotate_robot')

        # Default k value
        self.k = 2.0

        # Latest normalized input value
        self.x_value = 0.0

        # QoS profile for distributed communication
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.create_subscription(Float32, 'x_val/float', self.x_callback, qos)

        # Publisher
        self.publisher = self.create_publisher(Twist, '/cmd_vel', qos)

        # Timer
        self.timer = self.create_timer(0.1, self.publish_twist)


    def x_callback(self, msg):
        self.x_value = msg.data

    def publish_twist(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = - self.k * self.x_value
        self.publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = RotateRobot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("RotateRobot stopped cleanly (Ctrl+C).")
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher.publish(twist)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':

    main()




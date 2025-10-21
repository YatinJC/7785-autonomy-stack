#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
import numpy as np


class ControllerNode(Node):
    k1 = 2.  # Proportional gain (linear velocity)
    k2 = 5.  # Proportional gain (angular velocity)
    
    MAX_LINEAR_VELOCITY = .2
    MAX_ANGULAR_VELOCITY = 2.
    MIN_LINEAR_VELOCITY = -MAX_LINEAR_VELOCITY
    MIN_ANGULAR_VELOCITY = -MAX_ANGULAR_VELOCITY
    THRESHOLD_DIST = 0.05  # Stop if within 5cm of target

    def __init__(self):
        super().__init__('controller')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # Subscribe to corrected odometry
        self.create_subscription(Odometry, '/odom_corrected', self.odom_callback, 10)
        # Subscribe to the next location as a bare Point (x,y). The SearchNode publishes a Point.
        self.create_subscription(Point, '/next_location/point', self.point_callback, 10)

        # Current position and orientation from corrected odometry
        self.globalPos = Point()
        self.globalAng = 0.0
        self.target_point = None

        # Flag to log target only once per new target
        self.target_logged = False
     
        self.get_logger().info('Controller Node started')

    def odom_callback(self, msg):
        """
        Update position and orientation from corrected odometry and run control loop
        """
        # Get position directly from corrected odometry
        self.globalPos = msg.pose.pose.position

        # Extract yaw angle from quaternion
        q = msg.pose.pose.orientation
        self.globalAng = np.arctan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        
        if self.target_point is not None:
            # Run control loop
            self.control_loop()

    def point_callback(self, msg):
        # `msg` is a geometry_msgs/Point (x,y,z) published by SearchNode.
        # Assign directly instead of accessing `msg.point` which would be present
        # if this were a PointStamped or a custom message.
        self.target_point = msg
        # Reset flag when new target is received
        self.target_logged = False

    def control_loop(self):
        # Handle NaN target point (all goals reached)
        if math.isnan(self.target_point.x) or math.isnan(self.target_point.y):
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            # self.get_logger().info('All goals reached: stopping robot.')
            return
        # Get current position and orientation
        x = self.globalPos.x
        y = self.globalPos.y
        theta = self.globalAng

        # Target position
        tx = self.target_point.x
        ty = self.target_point.y

        # Log target position once when it's first received
        if not self.target_logged:
            self.get_logger().info(f'Target position: ({tx}, {ty})')
            self.target_logged = True
        # Compute error
        dx = tx - x
        dy = ty - y
        distance = math.sqrt(dx**2 + dy**2)
        target_theta = math.atan2(dy, dx)
        normalized_angle = self.normalize_angle(target_theta - theta)
        
        # Proportional control
        v = self.k1 * distance
        w = self.k2 * normalized_angle
        # Limit velocities
        v = max(min(v, self.MAX_LINEAR_VELOCITY), self.MIN_LINEAR_VELOCITY)
        w = max(min(w, self.MAX_ANGULAR_VELOCITY), self.MIN_ANGULAR_VELOCITY)
        # Publish Twist
        if normalized_angle > math.pi/4 or normalized_angle < -math.pi/4:
            v = 0.0
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)
        self.get_logger().info(f'Cmd published: linear.x={v:.2f}, angular.z={w:.2f}, distance to target={distance:.2f}, angle error={normalized_angle:.2f}')

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

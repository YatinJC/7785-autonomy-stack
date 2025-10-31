#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
import numpy as np


class ControllerNode(Node):
    # Pure Pursuit parameters
    LOOKAHEAD_DISTANCE = 0.25  # How far ahead to look on the path (meters)
    MIN_LOOKAHEAD = 0.15  # Minimum lookahead distance
    MAX_LOOKAHEAD = 0.4  # Maximum lookahead distance

    # Control parameters
    BASE_LINEAR_VELOCITY = 0.15  # Target forward velocity
    MAX_LINEAR_VELOCITY = 0.2
    MAX_ANGULAR_VELOCITY = 2.0
    MIN_LINEAR_VELOCITY = -MAX_LINEAR_VELOCITY
    MIN_ANGULAR_VELOCITY = -MAX_ANGULAR_VELOCITY

    # Angular velocity gain
    k_angular = 2.5

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

        self.get_logger().info('Controller Node started (Pure Pursuit)')

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
            self.get_logger().info(f'New target: ({tx:.2f}, {ty:.2f})')
            self.target_logged = True

        # Calculate distance to target
        dx = tx - x
        dy = ty - y
        distance = math.sqrt(dx**2 + dy**2)

        # Pure Pursuit Algorithm
        # 1. Calculate lookahead distance (adaptive based on distance to goal)
        lookahead = min(max(distance * 0.5, self.MIN_LOOKAHEAD), self.MAX_LOOKAHEAD)

        # 2. Find the lookahead point
        # For a single waypoint, use the target if we're close, otherwise use lookahead point
        if distance <= lookahead:
            # Close to target, aim directly at it
            lookahead_x = tx
            lookahead_y = ty
        else:
            # Use a point along the line from current position to target at lookahead distance
            direction = math.atan2(dy, dx)
            lookahead_x = x + lookahead * math.cos(direction)
            lookahead_y = y + lookahead * math.sin(direction)

        # 3. Transform lookahead point to robot frame
        dx_robot = lookahead_x - x
        dy_robot = lookahead_y - y

        # Rotate to robot's local frame
        dx_local = dx_robot * math.cos(-theta) - dy_robot * math.sin(-theta)
        dy_local = dx_robot * math.sin(-theta) + dy_robot * math.cos(-theta)

        # 4. Calculate curvature (steering) using Pure Pursuit formula
        # curvature = 2 * dy_local / lookahead^2
        lookahead_dist_actual = math.sqrt(dx_local**2 + dy_local**2)
        if lookahead_dist_actual < 0.01:
            curvature = 0.0
        else:
            curvature = 2.0 * dy_local / (lookahead_dist_actual ** 2)

        # 5. Calculate velocities
        # Linear velocity: constant when far, slow down when close
        if distance > 0.3:
            v = self.BASE_LINEAR_VELOCITY
        else:
            # Slow down smoothly as we approach
            v = self.BASE_LINEAR_VELOCITY * (distance / 0.3)
            v = max(v, 0.05)  # Minimum velocity to keep moving

        # Angular velocity: proportional to curvature
        w = curvature * v * self.k_angular

        # 6. Limit velocities
        v = max(min(v, self.MAX_LINEAR_VELOCITY), 0.0)  # No backward motion
        w = max(min(w, self.MAX_ANGULAR_VELOCITY), self.MIN_ANGULAR_VELOCITY)

        # 7. If target is behind us (angle > 90 degrees), just rotate in place
        angle_to_target = math.atan2(dy, dx)
        angle_diff = self.normalize_angle(angle_to_target - theta)
        if abs(angle_diff) > math.pi / 2:
            v = 0.0
            w = self.k_angular * angle_diff

        # Publish Twist
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)

        self.get_logger().info(
            f'Pure Pursuit: v={v:.2f}, w={w:.2f} | '
            f'dist={distance:.3f}m, lookahead={lookahead:.2f}m, curv={curvature:.2f}',
            throttle_duration_sec=0.5
        )

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

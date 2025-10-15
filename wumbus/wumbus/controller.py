#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
import numpy as np


class ControllerNode(Node):
    k1 = 1.0  # Proportional gain (linear velocity)
    k2 = 1.0  # Proportional gain (angular velocity)
    WHEEL_RADIUS = 0.033
    TURNING_RADIUS = 0.16
    MAX_LINEAR_VELOCITY = WHEEL_RADIUS * 2 * math.pi * 61 / 60
    MAX_ANGULAR_VELOCITY = MAX_LINEAR_VELOCITY / TURNING_RADIUS
    MIN_LINEAR_VELOCITY = -MAX_LINEAR_VELOCITY
    MIN_ANGULAR_VELOCITY = -MAX_ANGULAR_VELOCITY
    THRESHOLD_DIST = 0.05  # Stop if within 5cm of target

    def __init__(self):
        super().__init__('controller')
        self.current_pose = None
        self.target_point = None
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        # Subscribe to the next location as a bare Point (x,y). The SearchNode publishes a Point.
        self.create_subscription(Point, '/next_location/point', self.point_callback, 10)
        # Use a timer to run the control loop at a fixed rate (20Hz).
        # This decouples control updates from incoming message rates (odometry or goal updates)
        # so the controller can continue to compute and publish commands even if callbacks are
        # momentarily delayed or arrive at irregular intervals.
        self.timer = self.create_timer(0.05, self.control_loop)

        # Initialize odometry tracking variables
        self.Init = True
        self.Init_ang = 0.0
        self.Init_pos = Point()
        self.globalPos = Point()

    def update_Odometry(self,Odom):
        position = Odom.pose.pose.position
        
        #Orientation uses the quaternion aprametrization.
        #To get the angular position along the z-axis, the following equation is required.
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

        if self.Init:
            #The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
            self.Init_pos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Init_pos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Init_pos.z = position.z
        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        

        #We subtract the initial values
        self.globalPos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos.x
        self.globalPos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos.y
        self.globalAng = orientation - self.Init_ang
    

    def odom_callback(self, msg):
        self.update_Odometry(msg)

    def point_callback(self, msg):
        # `msg` is a geometry_msgs/Point (x,y,z) published by SearchNode.
        # Assign directly instead of accessing `msg.point` which would be present
        # if this were a PointStamped or a custom message.
        self.target_point = msg

    def control_loop(self):
        if self.current_pose is None or self.target_point is None:
            return
        # Handle NaN target point (all goals reached)
        if math.isnan(self.target_point.x) or math.isnan(self.target_point.y):
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            self.get_logger().info('All goals reached: stopping robot.')
            return
        # Get current position and orientation
        x = self.globalPos.x
        y = self.globalPos.y
        # Orientation as quaternion
        q = self.globalAng
        # Convert quaternion to yaw
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)
        # Target position
        tx = self.target_point.x
        ty = self.target_point.y
        # Compute error
        dx = tx - x
        dy = ty - y
        distance = math.sqrt(dx**2 + dy**2)
        target_theta = math.atan2(dy, dx)
        normalised_angle = self.normalize_angle(target_theta - theta)
        # Proportional control
        v = self.k1 * distance if distance > self.THRESHOLD_DIST else 0.0
        w = self.k2 * normalised_angle if abs(normalised_angle) > 0.05 else 0.0
        # Limit velocities
        v = max(min(v, self.MAX_LINEAR_VELOCITY), self.MIN_LINEAR_VELOCITY)
        w = max(min(w, self.MAX_ANGULAR_VELOCITY), self.MIN_ANGULAR_VELOCITY)
        # Publish Twist
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)

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

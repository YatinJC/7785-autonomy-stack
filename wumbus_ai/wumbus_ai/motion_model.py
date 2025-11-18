#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from std_msgs.msg import String


class MotionModelNode(Node):
    def __init__(self):
        super().__init__('motion_model_node')
        self.get_logger().info('Motion Model Node has been started.')

        # Initialize parameters
        self.max_lin_vel = 2.0  # meters per second
        self.max_ang_vel = 1.0  # radians per second

        # Wall following parameters and subscriptions

        self.boundary_threshold = 0.5  # meters
        self.gain_boundary = 1.0  # gain for following wall
        self.left_error = None
        self.right_error = None

        self.subscriber = self.create_subscription(Float32, '/left_dist/Float32', self.left_distance_callback, 10)
        self.subscriber = self.create_subscription(Float32, '/right_dist/Float32', self.right_distance_callback, 10)

        # Stopping condition parameters and subscription

        self.stop_flag = False
        self.stop_distance_threshold = 0.3  # meters
        self.subscriber = self.create_subscription(Float32, '/forward_dist/Float32', self.forward_distance_callback, 10)

        self.vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # sign function
        self.create_subscription(String, '/Sign/String', self.sign_callback, 1)

        self.sign_dectected_flag = False



# Follow Wall and stop functinonality

    def left_distance_callback(self, msg):
        left_distance = msg.data
        self.left_error = self.boundary_threshold - left_distance

        self.velocity_publisher()
    
    def right_distance_callback(self, msg):
        right_distance = msg.data
        self.right_error = self.boundary_threshold - right_distance

        self.velocity_publisher()

    def forward_distance_callback(self, msg):
        forward_distance = msg.data
        forward_error = self.stop_distance_threshold - forward_distance

        if forward_error < 0:
            self.stop_flag = True
        else:
            self.stop_flag = False

        self.velocity_publisher()

    def lost_behavior(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = self.max_ang_vel  # Rotate in place
        return twist


# Velocity publisher function

    def velocity_publisher(self):
        if not self.sign_dectected_flag:
            linear_velocity = self.max_lin_vel
            angular_velocity = 0.0

            if not self.stop_flag:
                if self.left_error is not None:
                    angular_velocity = self.gain_boundary * self.left_error
                elif self.right_error is not None:
                    angular_velocity = -self.gain_boundary * self.right_error
                else:
                    twist = self.lost_behavior()
                    self.get_logger().info('Lost: executing lost behavior.')
                    self.vel_publisher.publish(twist)
                    return  # Exit to avoid publishing conflicting command after lost behavior
                    

                # Clamp angular velocity to max limit
                angular_velocity = max(min(angular_velocity, self.max_ang_vel), -self.max_ang_vel)

                twist = Twist()
                twist.linear.x = linear_velocity
                twist.angular.z = angular_velocity

                self.vel_publisher.publish(twist)
            else:
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.get_logger().info('Stopping: forward obstacle detected.')
                self.vel_publisher.publish(twist)


# Sign Funcitonalities:
    def sign_callback(self, msg):
        self.sign_dectected_flag = True
        sign = msg.data

        if hasattr(self, "timer") and self.timer is not None:
            self.timer.cancel()
            self.timer = None

        twist = Twist()
        twist.linear.x = 0.0
        duration = None # Initialize duration variable

        if sign == "L":
           twist.angular.z = self.max_ang_vel
           duration = 1.57 / self.max_ang_vel  # 90 degrees turn to left
           self.get_logger().info('Left Sign Detected: executing left turn.')
        
        elif sign == "R":
            twist.angular.z = -self.max_ang_vel
            duration = 1.57 / self.max_ang_vel  # 90 degrees turn to right
            self.get_logger().info('Right Sign Detected: executing right turn.')

        elif sign == "D":
            twist.angular.z = self.max_ang_vel
            duration = 3.14 / self.max_ang_vel  # 180 degrees turn
            self.get_logger().info('Dead End Sign Detected: executing U-turn.')

        elif sign == "S":
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info('Stop Sign Detected: stopping robot.')

        elif sign == "G":
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info('Goal Sign Detected: stopping robot.')
            rclpy.shutdown()

        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info('Unknown Sign Detected: stopping robot.')


        self.vel_publisher.publish(twist)
        if duration is not None:
            self.timer = self.create_timer(duration, self.stop_after_turn)

    def stop_after_turn(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.vel_publisher.publish(twist)
        self.get_logger().info('Turn completed: stopping robot.')
        self.sign_dectected_flag = False
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None


def main():
    rclpy.init()
    node = MotionModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()







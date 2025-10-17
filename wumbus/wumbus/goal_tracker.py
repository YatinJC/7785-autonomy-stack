#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import math
import numpy as np

class GoalTracker(Node):
    def __init__(self):
        super().__init__('goal_tracker')
        self.flag = True
        # Publisher for current goal (2D point)
        self.goal_pub = self.create_publisher(Point, '/current_goal/Point', 10)

        # Subscriber to corrected odometry
        self.odom_sub = self.create_subscription(Odometry, '/odom_corrected', self.odom_callback, 10)

        # Define 3 goals (x, y)
        self.goals = [
            Point(x=1.5, y=0.0, z=0.0),
            Point(x=1.5, y=1.4, z=0.0),
            Point(x=0.0, y=1.4, z=0.0)
        ]

        self.goal_index = 0  # start with first goal
        self.threshold = 0.05  # Stop if within 5cm of target
        self.current_pose = Point()

        # Publish the first goal
        self.publish_current_goal()
        self.get_logger().info('Goal Tracker Node started')

    def odom_callback(self, msg: Odometry):
        """
        Update current position from corrected odometry and check goal distance
        """
        # Get position directly from corrected odometry
        self.current_pose = msg.pose.pose.position

        current_goal = self.goals[self.goal_index]

        # Calculate distance to goal
        distance = math.sqrt(
            (current_goal.x - self.current_pose.x) ** 2 +
            (current_goal.y - self.current_pose.y) ** 2
        )

        if distance < self.threshold:
            self.get_logger().info(f'Goal {self.goal_index + 1} reached')
            self.switch_goal()

        self.publish_current_goal()
        if self.flag == True:
            self.get_logger().info(f'Current Position: ({self.current_pose.x:.2f}, {self.current_pose.y:.2f}), Current Goal: ({current_goal.x}, {current_goal.y}), Distance: {distance:.2f}')
            self.flag = False
   
    def switch_goal(self):
        # Move to the next goal if available
        if self.goal_index < len(self.goals) - 1:
            self.goal_index += 1
            self.flag = True
        
        else:
            self.get_logger().info('All goals reached!')
            # Publish None as current goal (use NaN values)
            import math
            none_goal = Point(x=math.nan, y=math.nan, z=math.nan)
            self.goal_pub.publish(none_goal)
            # Optionally, you could stop the node or publish a special message here

    def publish_current_goal(self):
        goal = self.goals[self.goal_index]
        self.goal_pub.publish(goal)
        # No logging for publishing goal

def main(args=None):
    rclpy.init(args=args)
    node = GoalTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

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

        # Publisher for current goal (2D point)
        self.goal_pub = self.create_publisher(Point, '/current_goal/Point', 10)

        # Subscriber to odometry
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Define 3 goals (x, y)
        self.goals = [
            Point(x=1.5, y=0.0, z=0.0),
            Point(x=1.5, y=1.4, z=0.0),
            Point(x=0.0, y=1.4, z=0.0)
        ]

        self.goal_index = 0  # start with first goal
        self.threshold = 0.05  # Stop if within 5cm of target
        self.current_pose = Point()

        # Initialize odometry tracking variables
        self.Init = True
        self.Init_ang = 0.0
        self.Init_pos = Point()
        self.globalPos = Point()

        # Publish the first goal
        self.publish_current_goal()

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
        self.current_pose.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos.x
        self.current_pose.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos.y
        self.globalAng = orientation - self.Init_ang

    def odom_callback(self, msg: Odometry):
        self.update_Odometry(msg)

        current_goal = self.goals[self.goal_index]

        # Calculate distance to goal
        distance = math.sqrt(
            (current_goal.x - self.current_pose.x) ** 2 +
            (current_goal.y - self.current_pose.y) ** 2
        )

        if distance < self.threshold:
            self.get_logger().info(f'Goal {self.goal_index + 1} reached')
            self.switch_goal()

    def switch_goal(self):
        # Move to the next goal if available
        if self.goal_index < len(self.goals) - 1:
            self.goal_index += 1
            self.publish_current_goal()
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

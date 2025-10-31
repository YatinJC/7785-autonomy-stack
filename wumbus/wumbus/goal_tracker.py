#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import math
import numpy as np

class GoalTracker(Node):
    def __init__(self):
        super().__init__('goal_tracker')
        self.flag = True
        # Publisher for current goal (2D point)
        self.goal_pub = self.create_publisher(Point, '/current_goal/Point', 10)

        # Publisher for goal visualization
        self.goal_viz_pub = self.create_publisher(MarkerArray, '/goal_markers', 10)

        # Subscriber to corrected odometry
        self.odom_sub = self.create_subscription(Odometry, '/odom_corrected', self.odom_callback, 10)

        # Subscriber to clicked points from RViz
        self.clicked_point_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_point_callback,
            10)

        # Start with empty waypoint list (can be dynamically added)
        self.goals = []

        self.goal_index = 0  # start with first goal
        self.threshold = 0.05  # Stop if within 5cm of target
        self.current_pose = Point()
        self.current_goal_reached = True  # No goals yet

        self.get_logger().info('Goal Tracker Node started - waiting for waypoints via /clicked_point')

    def clicked_point_callback(self, msg: PointStamped):
        """
        Add a new waypoint from RViz clicked point
        """
        new_goal = Point(x=msg.point.x, y=msg.point.y, z=0.0)
        self.goals.append(new_goal)
        self.get_logger().info(f'Added new waypoint: ({msg.point.x:.2f}, {msg.point.y:.2f})')
        self.get_logger().info(f'Total waypoints: {[(round(g.x, 2), round(g.y, 2)) for g in self.goals]}')

        # Update visualization
        self.publish_goal_markers()

        # If this is the first goal and we're not currently tracking one, start publishing
        if len(self.goals) == 1 and self.current_goal_reached:
            self.current_goal_reached = False
            self.publish_current_goal()

    def odom_callback(self, msg: Odometry):
        """
        Update current position from corrected odometry and check goal distance
        """
        # Get position directly from corrected odometry
        self.current_pose = msg.pose.pose.position

        # Only check goals if we have any and haven't reached all of them
        if len(self.goals) == 0 or self.goal_index >= len(self.goals):
            return

        current_goal = self.goals[self.goal_index]

        # Calculate distance to goal
        distance = math.sqrt(
            (current_goal.x - self.current_pose.x) ** 2 +
            (current_goal.y - self.current_pose.y) ** 2
        )

        if distance < self.threshold:
            self.get_logger().info(f'Goal {self.goal_index + 1} reached')
            self.switch_goal()
            # Update visualization when goal changes
            self.publish_goal_markers()
        else:
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
        if self.goal_index < len(self.goals):
            goal = self.goals[self.goal_index]
            self.goal_pub.publish(goal)
            # No logging for publishing goal

    def publish_goal_markers(self):
        """
        Publish visualization markers for all goals in RViz
        """
        marker_array = MarkerArray()

        # Clear all previous markers first
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        self.goal_viz_pub.publish(marker_array)

        # Create new marker array
        marker_array = MarkerArray()

        for i, goal in enumerate(self.goals):
            # Create sphere marker for each goal
            marker = Marker()
            marker.header.frame_id = "odom_corrected"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "goals"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Set position
            marker.pose.position.x = goal.x
            marker.pose.position.y = goal.y
            marker.pose.position.z = 0.1  # Slightly above ground
            marker.pose.orientation.w = 1.0

            # Set scale
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.15

            # Set color based on status
            if i < self.goal_index:
                # Completed goals - gray
                marker.color.r = 0.5
                marker.color.g = 0.5
                marker.color.b = 0.5
                marker.color.a = 0.5
            elif i == self.goal_index:
                # Current goal - green
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            else:
                # Future goals - yellow
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.8

            marker.lifetime.sec = 0  # Persistent marker

            marker_array.markers.append(marker)

            # Add text label
            text_marker = Marker()
            text_marker.header.frame_id = "odom_corrected"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "goal_labels"
            text_marker.id = i + 1000  # Offset ID to avoid collision
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = goal.x
            text_marker.pose.position.y = goal.y
            text_marker.pose.position.z = 0.3  # Above the sphere
            text_marker.pose.orientation.w = 1.0

            text_marker.scale.z = 0.1  # Text height
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            text_marker.text = f"Goal {i + 1}"
            text_marker.lifetime.sec = 0

            marker_array.markers.append(text_marker)

        self.goal_viz_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = GoalTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

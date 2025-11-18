#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PointStamped, PoseStamped

class Nav2ActionClient(Node):
    def __init__(self):
        super().__init__('nav2_action_client')
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.action_client.wait_for_server()
        
        self.subscriber = self.create_subscription(PointStamped, '/clicked_point', self.waypoints_callback, 10)
        self.waypoints = []

        self.get_logger().info('Nav2 Action Client Node has been started.')

        self.pass_points = False



        
    def waypoints_callback(self, msg):

        if len(self.waypoints) >= 3:
            self.get_logger().info('Maximum of 3 waypoints reached. Ignoring additional waypoints.')
            return
        x = msg.point.x
        y = msg.point.y
        self.get_logger().info(f'Received waypoint: x={x}, y={y}')
        self.waypoints.append((x, y))

        self.get_logger().info(f'Total waypoints received: {len(self.waypoints)}')

        if not self.pass_points and len(self.waypoints) == 3:
            self.pass_points = True
            self.send_goal()

    def send_goal(self):

        if not self.waypoints:
            self.get_logger().info('No waypoints to send.')
            return
        
        x, y = self.waypoints.pop(0)

        goal_msg = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0  # Neutral orientation

        goal_msg.pose = pose



        self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback).add_done_callback(self.goal_response_callback)

        
    def feedback_callback(self, fb):
        pass  # Feedback handling can be implemented here if needed
        

    def goal_response_callback(self, future_response):
        goal_handle = future_response.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal accepted.')
            return

        self.get_logger().info('Goal accepted.')
        goal_handle.get_result_async().add_done_callback(self.goal_result_callback)


    def goal_result_callback(self, future_result):
        result = future_result.result()
        if result and result.status == 4:  # SUCCEEDED this is from nav2_msgs/NavigationResult
            self.get_logger().info('Goal accepted.')
            self.send_goal()
        else:
            self.get_logger().info('Goal rejected.') 
        

def main(args=None):
    try:
        rclpy.init(args=args)
        action_client = Nav2ActionClient()
        rclpy.spin(action_client)
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        action_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
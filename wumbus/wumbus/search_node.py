#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import struct
import numpy as np
import math

class SearchNode(Node):
    def __init__(self):
        super().__init__('search_node')
        self.resolution = self.get_parameter('resolution').get_parameter_value().double_value
        self.flag = True
        self.current_location = None
        self.goal_location = None
        self.obstacles = set()


        # Initialize odometry tracking variables
        self.Init = True
        self.Init_ang = 0.0
        self.Init_pos = Point()
        self.globalPos = Point()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription_odom = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.subscription_goal = self.create_subscription(
            Point,
            '/current_goal/Point',
            self.goal_callback,
            10)
        # Subscribe to obstacle points published as a PointCloud2
        self.subscription_obstacles = self.create_subscription(
            PointCloud2,
            '/obstacle_points/PointCloud2',
            self.obstacles_callback,
            qos)
            
        self.publisher_next = self.create_publisher(
            Point,
            '/next_location/point',
            10)
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
        # Round odometry to 1 decimal place as requested to match obstacle map scaling
        self.update_Odometry(msg)
        x = round(self.globalPos.x, 1)
        y = round(self.globalPos.y, 1)
        self.current_location = (x, y)
        self.plan_and_publish()

    def goal_callback(self, msg):
        # Check for NaN goal (all goals reached)
        if math.isnan(msg.x) or math.isnan(msg.y):
            self.goal_location = None
            self.get_logger().info('All goals reached: stopping path planning.')
            # Publish NaN to /next_location/point so controller can stop
            none_point = Point(x=math.nan, y=math.nan, z=math.nan)
            self.publisher_next.publish(none_point)
            return
        self.goal_location = (msg.x, msg.y)
        

    def obstacles_callback(self, msg):
        """
        Convert PointCloud2 message to a set of (x, y) tuples
        """

        # Extract point data from the PointCloud2 message
        # The message has fields x, y, z (each 4 bytes, float32)
        point_step = msg.point_step  # Should be 12 bytes (3 floats)

        # Iterate through the point cloud data
        new_obs = set()
        for i in range(msg.width):
            # Calculate the byte offset for this point
            offset = i * point_step

            # Unpack x, y, z values (we only need x and y)
            x, y, z = struct.unpack_from('fff', msg.data, offset)

            # Add (x, y) tuple to the set
            new_obs.add((round(x,1), round(y,1)))
        self.obstacles = new_obs

        
        # if self.flag == True:
        #     self.get_logger().info(f'Obstacles: {self.obstacles}')
        #     self.flag = False


    def plan_and_publish(self):
        if self.current_location and self.goal_location and self.obstacles is not None:
            path = self.a_star(self.current_location, self.goal_location, self.obstacles)
            if len(path) > 1:
                next_point = path[1]  # Next step
                point_msg = Point()
                point_msg.x, point_msg.y = next_point[0], next_point[1]
                point_msg.z = 0.0
                self.publisher_next.publish(point_msg)
                self.get_logger().info(f'Next point published: ({point_msg.x}, {point_msg.y})')

    def a_star(self, start, goal, obstacles):
        from heapq import heappush, heappop
        resolution = .1
        def neighbors(node):
            resolution = .1
            x, y = node
            moves = [
                (resolution, 0, 1 * resolution ** 2),
                (-resolution, 0, 1 * resolution ** 2),
                (0, resolution, 1 * resolution ** 2),
                (0, -resolution, 1 * resolution ** 2),
                (resolution, resolution, 2 * resolution ** 2),
                (resolution, -resolution, 2 * resolution ** 2),
                (-resolution, resolution, 2 * resolution ** 2),
                (-resolution, -resolution, 2 * resolution ** 2),
            ]
            for dx, dy, cost in moves:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in obstacles:
                    yield (nx, ny), cost
        def heuristic(a, b):
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

        open_set = []
        heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))
        closed_set = set()
        while open_set:
            f, g, current, path = heappop(open_set)
            if current in closed_set:
                continue
            if heuristic(current, goal) < resolution ** 2:
                return path + [goal]
            closed_set.add(current)
            for neighbor, move_cost in neighbors(current):
                if neighbor in closed_set:
                    continue
                heappush(open_set, (g + move_cost + heuristic(neighbor, goal), g + move_cost, neighbor, path + [neighbor]))
        return None

def main(args=None):
    rclpy.init(args=args)
    node = SearchNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

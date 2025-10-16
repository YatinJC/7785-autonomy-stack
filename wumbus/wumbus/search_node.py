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
import heapq

class SearchNode(Node):
    def __init__(self):
        super().__init__('search_node')
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
        x = round(self.globalPos.x,1)
        y = round(self.globalPos.y,1)
        self.current_location = (x, y)
        self.plan_and_publish()

    def goal_callback(self, msg):
        # Check for NaN goal (all goals reached)
        if math.isnan(msg.x) or math.isnan(msg.y):
            # self.goal_location = None
            # self.get_logger().info('All goals reached: stopping path planning.')
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
        # The message has fields x, y (each 4 bytes, float32)
        # Note: derez_lidar packs 3 floats (x, y, z=0.0) but only defines 2 fields
        # So we need to unpack based on the actual packed data (12 bytes per point)
        point_step = 8  # 3 floats * 4 bytes each (x, y, z)

        # Iterate through the point cloud data
        new_obs = set()
        for i in range(msg.width):
            # Calculate the byte offset for this point
            offset = i * point_step

            # Unpack x, y, z values (we only need x and y)
            x, y = struct.unpack_from('ff', msg.data, offset)

            # Add (x, y) tuple to the set
            new_obs.add((round(x, 1), round(y, 1)))
        self.obstacles = new_obs

        
        # if self.flag == True:
        self.get_logger().info(f'Obstacles: {self.obstacles}')
            # self.flag = False


    def plan_and_publish(self):

        if self.current_location and self.goal_location and self.obstacles is not None:
            path = self.astar(self.current_location, self.goal_location, self.obstacles, resolution=0.1)
            self.get_logger().info(f'Current: {self.current_location}, Goal: {self.goal_location}, Path: {path}')
            
            if path is None:
                self.get_logger().warn('No path found to goal!')
                self.publisher_next.publish(Point(x=0., y=0., z=0.0))
                return
                
            if len(path) > 1:
                next_point = path[1]  # Next point in the path
                point_msg = Point()
                point_msg.x, point_msg.y = round(next_point[0],1), round(next_point[1],1)
                point_msg.z = 0.0
                self.publisher_next.publish(point_msg)
                self.get_logger().info(f'Next point published: ({point_msg.x}, {point_msg.y})')

    def astar(self, start, goal, obstacles, resolution):
        """
        A* path planner on an 8-connected grid.

        Inputs:
        - start: (x, y) tuple in world coordinates (floats)
        - goal: (x, y) tuple in world coordinates (floats)
        - obstacles: set or iterable of (x, y) world coordinates representing blocked points
        - resolution: grid cell size (float). Grid coordinates are snapped to multiples of this.

        Returns:
        - path: list of (x, y) world coordinates from start to goal (start included). If no path,
        raises ValueError.

        Notes:
        - This implementation quantizes the world into a grid using "resolution" to avoid floating-point
        hashing issues and to ensure proper A* optimality guarantees on the grid graph.
        - Diagonal corner-cutting around obstacles is prevented.
        """

        # --- Helpers: grid/world conversions and costs ---
        def to_grid(p):
            """Convert world coordinate (float, float) to integer grid cell (i, j)."""
            return (int(round(p[0] / resolution)), int(round(p[1] / resolution)))

        def to_world(c):
            """Convert integer grid cell (i, j) to world coordinate (x, y)."""
            return (round(c[0] * resolution,1), round(c[1] * resolution,1))

        def step_cost(a, b):
            """Cost between adjacent grid cells a->b (cardinal: res, diagonal: res*sqrt(2))."""
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            if dx + dy == 1:
                return resolution
            # diagonal
            return math.sqrt(2) * resolution

        def heuristic(c):
            """Consistent Euclidean heuristic in grid space scaled by resolution."""
            dx = c[0] - goal_c[0]
            dy = c[1] - goal_c[1]
            return math.hypot(dx, dy) * resolution

        # --- Inflate obstacles in grid space ---
        def inflate_obstacles_grid(obstacles_w):
            """Inflate obstacles by 1 cell in Chebyshev radius in grid space."""
            avoid_cells = set()
            for (x, y) in obstacles_w:
                cx, cy = to_grid((x, y))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        n = (cx + dx, cy + dy)
                        if n == goal_c:
                            continue
                        avoid_cells.add(n)
            return avoid_cells

        # --- Neighbor generation with corner-cutting prevention ---
        def neighbors(c, avoid):
            x, y = c
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    n = (nx, ny)
                    if n in avoid:
                        continue
                    # prevent diagonal corner cutting: require the orthogonal cells to be free
                    if dx != 0 and dy != 0:
                        if (x + dx, y) in avoid or (x, y + dy) in avoid:
                            continue
                    yield n

        # Early exit
        if start == goal:
            return [start]

        # Convert to grid
        start_c = to_grid(start)
        goal_c = to_grid(goal)

        avoid = inflate_obstacles_grid(obstacles)
        # Ensure start/goal are traversable even if inside inflated region
        if start_c in avoid:
            avoid.remove(start_c)

        # A* structures
        open_heap = []  # entries: (f, h, (ix, iy))
        heapq.heappush(open_heap, (heuristic(start_c), heuristic(start_c), start_c))
        came_from = {}
        g_score = {start_c: 0.0}
        closed = set()

        while open_heap:
            f, h, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal_c:
                # reconstruct path (grid -> world)
                grid_path = [current]
                while current in came_from:
                    current = came_from[current]
                    grid_path.append(current)
                grid_path.reverse()
                world_path = [to_world(c) for c in grid_path]
                # ensure exact start/goal endpoints for nicety
                if world_path[0] != start:
                    world_path[0] = start
                if world_path[-1] != goal:
                    world_path.append(goal)
                return world_path

            closed.add(current)

            for n in neighbors(current, avoid):
                if n in closed:
                    continue
                tentative_g = g_score[current] + step_cost(current, n)
                if tentative_g < g_score.get(n, float("inf")):
                    came_from[n] = current
                    g_score[n] = tentative_g
                    hn = heuristic(n)
                    heapq.heappush(open_heap, (tentative_g + hn, hn, n))

        return None

def main(args=None):
    rclpy.init(args=args)
    node = SearchNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

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
import time

class SearchNode(Node):
    WAYPOINT_THRESHOLD = 0.05  # Distance threshold to consider waypoint reached

    def __init__(self):
        super().__init__('search_node')
        self.flag = True
        self.current_location = None
        self.goal_location = None
        self.obstacles = set()

        # Path tracking
        self.current_path = None  # List of waypoints from current position to goal
        self.current_waypoint_index = 0  # Index of the next waypoint to reach
        self.is_planning = False  # Flag to indicate if path planning is in progress

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        
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
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom_corrected',
            self.odom_callback,
            10
        )
        self.publisher_next = self.create_publisher(
            Point,
            '/next_location/point',
            10)
        self.get_logger().info('Search Node started')

    def odom_callback(self, msg):
        """
        Update current location from corrected odometry and check/advance waypoints
        """
        position = msg.pose.pose.position
        self.current_location = (round(position.x, 1), round(position.y, 1))
        self.get_logger().info(f'Current location updated: {self.current_location}')
        # Check if we need to advance to the next waypoint or replan
        if self.current_path and self.goal_location:
            # Check if current waypoint is reached
            if self.current_waypoint_index < len(self.current_path):
                current_waypoint = self.current_path[self.current_waypoint_index]
                distance = math.hypot(
                    position.x - current_waypoint[0],
                    position.y - current_waypoint[1]
                )

                if distance < self.WAYPOINT_THRESHOLD:
                    # Waypoint reached, advance to next
                    self.current_waypoint_index += 1
                    self.get_logger().info(f'Waypoint {self.current_waypoint_index - 1} reached')

                    # Publish next waypoint
                    if self.current_waypoint_index < len(self.current_path):
                        next_waypoint = self.current_path[self.current_waypoint_index]
                        point_msg = Point()
                        point_msg.x, point_msg.y = next_waypoint[0], next_waypoint[1]
                        point_msg.z = 0.0
                        self.publisher_next.publish(point_msg)
                        self.get_logger().info(f'Next waypoint published: ({point_msg.x}, {point_msg.y})')
                    else:
                        # Goal reached
                        self.get_logger().info('Goal reached!')
                        self.current_path = None
                        self.current_waypoint_index = 0

    def goal_callback(self, msg):
        # Check for NaN goal (all goals reached)
        if math.isnan(msg.x) or math.isnan(msg.y):
            # Publish NaN to /next_location/point so controller can stop
            none_point = Point(x=math.nan, y=math.nan, z=math.nan)
            self.publisher_next.publish(none_point)
            self.current_path = None
            self.goal_location = None
            return
        if (round(msg.x,1), round(msg.y,1)) == self.goal_location:
            # self.get_logger().info('Same goal as before, ignoring.')
            return  # Same goal as before, ignore
        
        if self.current_location is None:
            self.get_logger().warn('Current location unknown, cannot plan path to goal yet.')
            return
        
        self.goal_location = (round(msg.x, 1), round(msg.y, 1))
        self.get_logger().info(f'New goal received: {self.goal_location}')

        # Stop the robot for 2 seconds before planning
        if self.current_location:
            self.stop_robot()
            self.get_logger().info('Stopping for 2 seconds before planning path to new goal...')
            time.sleep(2.0)
            # Plan initial path to the goal
            self.plan_path()
            self.get_logger().info('Path planning to new goal completed.')
        

    def obstacles_callback(self, msg):
        """
        Convert PointCloud2 message to a set of (x, y) tuples and check for path collisions
        """
        point_step = 8  # 2 floats * 4 bytes each (x, y)

        # Iterate through the point cloud data
        new_obs = set()
        for i in range(msg.width):
            # Calculate the byte offset for this point
            offset = i * point_step

            # Unpack x, y values
            x, y = struct.unpack_from('ff', msg.data, offset)

            # Add (x, y) tuple to the set
            new_obs.add((round(x, 1), round(y, 1)))

        self.obstacles = new_obs

        # Check if existing path collides with new obstacles
        if self.current_path and self.goal_location and not self.is_planning:
            if self.path_has_collision():
                self.get_logger().warn('Path collision detected! Stopping robot and replanning...')
                # Stop the robot immediately
                self.stop_robot()
                # Start replanning
                self.plan_path()


    def path_has_collision(self):
        """
        Check if the remaining path (from current waypoint onwards) collides with obstacles.
        Uses the same inflation logic as A* for consistency.
        """
        if not self.current_path or self.current_waypoint_index >= len(self.current_path):
            return False

        # Get inflated obstacles (same as in A* planning)
        resolution = 0.1

        def to_grid(p):
            return (int(round(p[0] / resolution)), int(round(p[1] / resolution)))

        # Inflate obstacles by 1 cell in grid space
        inflated_obstacles = set()
        for (x, y) in self.obstacles:
            cx, cy = to_grid((x, y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    inflated_obstacles.add((cx + dx, cy + dy))

        # Check remaining path points
        for i in range(self.current_waypoint_index, len(self.current_path)):
            waypoint = self.current_path[i]
            grid_pos = to_grid(waypoint)

            # Check if waypoint is in inflated obstacle area
            if grid_pos in inflated_obstacles:
                # Don't consider goal collision to allow reaching goal even if near obstacle
                if waypoint == self.goal_location:
                    continue
                return True

        return False

    def stop_robot(self):
        """
        Publish current location as target to make the robot stop
        """
        if self.current_location:
            stop_point = Point()
            stop_point.x, stop_point.y = math.nan, math.nan
            stop_point.z = math.nan
            self.publisher_next.publish(stop_point)
            self.get_logger().info('Robot stopped for replanning')

    def plan_path(self):
        """
        Plan a path from current location to goal using A* and publish first waypoint
        """
        if not self.current_location or not self.goal_location:
            return

        # Set planning flag to prevent concurrent replanning
        self.is_planning = True
        self.get_logger().info(f'Path planning started: Current={self.current_location}, Goal={self.goal_location}')

        path = self.astar(self.current_location, self.goal_location, self.obstacles, resolution=0.1)

        if path is None:
            self.get_logger().warn('No path found to goal!')
            self.current_path = None
            self.current_waypoint_index = 0
            self.is_planning = False
            return

        self.current_path = path
        self.current_waypoint_index = 0
        self.get_logger().info(f'Path found with {len(path)} waypoints')

        # Publish first waypoint (skip index 0 as it's the current location)
        if len(path) > 1:
            self.current_waypoint_index = 1
            next_point = path[1]
            point_msg = Point()
            point_msg.x, point_msg.y = next_point[0], next_point[1]
            point_msg.z = 0.0
            self.publisher_next.publish(point_msg)
            self.get_logger().info(f'First waypoint published: ({point_msg.x}, {point_msg.y}) - Robot resuming motion')

        # Clear planning flag
        self.is_planning = False

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

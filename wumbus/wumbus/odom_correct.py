#!/usr/bin/env python3
"""
Odometry Correction Node for TurtleBot3 Autonomy Stack

This node combines wheel odometry with LiDAR-based scan matching (ICP) to provide
more accurate pose estimation. It corrects for wheel slip, drift, and initial position
offset, publishing improved odometry to /odom_corrected.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
import numpy as np
import struct
from collections import deque


class OdometryCorrectNode(Node):
    """
    ROS 2 node that corrects odometry using both wheel encoder data and LiDAR scan matching.

    Subscribes to:
        /odom (Odometry): Raw wheel odometry from encoders
        /derez_lidar_base/PointCloud2 (PointCloud2): Transformed LiDAR point cloud

    Publishes:
        /odom_corrected (Odometry): Corrected odometry combining wheel and ICP estimates

    The node performs two levels of correction:
    1. Zeros out initial position and orientation (starts at origin)
    2. Applies ICP (Iterative Closest Point) on consecutive LiDAR scans to correct
       for wheel slip and drift, providing more accurate localization
    """

    def __init__(self):
        super().__init__('odom_correct')

        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to raw odometry
        self.subscription_odom = self.create_subscription(
            Odometry,
            '/odom',
            self.update_Odometry,
            10)

        # Subscribe to base transformed lidar scan
        self.subscription_obstacles = self.create_subscription(
            PointCloud2,
            '/derez_lidar_base/PointCloud2',
            self.derez_callback,
            sensor_qos)

        # Publish corrected odometry
        self.odom_pub = self.create_publisher(Odometry, '/odom_corrected', 10)

        # Initialize position tracking variables for wheel odometry
        self.Init = True
        self.Init_ang = 0.0
        self.globalAng = 0.0
        self.Init_pos = Point()
        self.globalPos = Point()

        # Initialize variables for ICP-based correction
        self.prev_lidar = None  # Store previous point cloud for ICP
        self.icp_correction_x = 0.0  # Cumulative ICP correction in x
        self.icp_correction_y = 0.0  # Cumulative ICP correction in y
        self.icp_correction_theta = 0.0  # Cumulative ICP correction in orientation

        # Odometry deltas for motion compensation
        self.delta_x_odom = 0.0
        self.delta_y_odom = 0.0
        self.delta_yaw_odom = 0.0

        # Buffer to store recent raw odometry messages for timestamp synchronization
        # Keep last 100 messages (should cover ~1 second at 100Hz odometry)
        self.odom_buffer = deque(maxlen=100)

        self.get_logger().info('Odometry Correction Node started')

    def get_synced_odom(self, target_timestamp):
        """
        Find the raw odometry message closest to the target timestamp.

        Args:
            target_timestamp (Time): Target timestamp to match

        Returns:
            Odometry or None: Closest odometry message, or None if buffer is empty
        """
        if not self.odom_buffer:
            return None

        # Convert target timestamp to nanoseconds for comparison
        target_ns = target_timestamp.sec * 1e9 + target_timestamp.nanosec

        # Find odometry message with closest timestamp
        closest_odom = None
        min_time_diff = float('inf')

        for odom_msg in self.odom_buffer:
            odom_ns = odom_msg.header.stamp.sec * 1e9 + odom_msg.header.stamp.nanosec
            time_diff = abs(odom_ns - target_ns)

            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_odom = odom_msg

        # Log if time difference is large (>50ms)
        if min_time_diff > 50e6:  # 50 milliseconds in nanoseconds
            self.get_logger().warn(
                f'Large time sync difference: {min_time_diff / 1e6:.1f}ms'
            )

        return closest_odom

    def parse_pointcloud2(self, msg: PointCloud2):
        """
        Parse PointCloud2 message and extract (x, y) coordinates.

        Args:
            msg (PointCloud2): Input point cloud message

        Returns:
            np.ndarray: Nx2 array of (x, y) coordinates
        """
        points = []
        point_step = msg.point_step

        for i in range(msg.width):
            offset = i * point_step
            # Unpack x and y (two floats)
            x, y = struct.unpack_from('ff', msg.data, offset)
            points.append([x, y])

        return np.array(points)

    def icp(self, source, target, max_iterations=20, tolerance=1e-5):
        """
        Iterative Closest Point (ICP) algorithm for 2D point cloud alignment.

        Args:
            source (np.ndarray): Nx2 source point cloud
            target (np.ndarray): Mx2 target point cloud
            max_iterations (int): Maximum number of iterations
            tolerance (float): Convergence threshold

        Returns:
            tuple: (R, t, mean_error) where R is 2x2 rotation matrix,
                   t is 2x1 translation vector, and mean_error is final alignment error
        """
        mean_error = 0.0
        
        if len(source) < 3 or len(target) < 3:
            # Not enough points for ICP
            return np.eye(2), np.zeros(2), float('inf')

        src = source.copy()
        prev_error = float('inf')

        R = np.eye(2)  # Initialize rotation matrix
        t = np.zeros(2)  # Initialize translation vector

        for iteration in range(max_iterations):
            # Find nearest neighbors
            distances = np.zeros(len(src))
            indices = np.zeros(len(src), dtype=int)

            for i, point in enumerate(src):
                # Calculate distances to all target points
                dists = np.sum((target - point) ** 2, axis=1)
                indices[i] = np.argmin(dists)
                distances[i] = dists[indices[i]]

            # Calculate mean error
            mean_error = np.mean(distances)

            # Check for convergence
            if abs(prev_error - mean_error) < tolerance:
                break

            prev_error = mean_error

            # Get corresponding points
            matched_target = target[indices]

            # Compute centroids
            centroid_src = np.mean(src, axis=0)
            centroid_tgt = np.mean(matched_target, axis=0)

            # Center the point clouds
            src_centered = src - centroid_src
            tgt_centered = matched_target - centroid_tgt

            # Compute rotation using SVD
            H = src_centered.T @ tgt_centered
            U, _, Vt = np.linalg.svd(H)
            R_iter = Vt.T @ U.T

            # Ensure proper rotation (det = 1)
            if np.linalg.det(R_iter) < 0:
                Vt[-1, :] *= -1
                R_iter = Vt.T @ U.T

            # Compute translation
            t_iter = centroid_tgt - R_iter @ centroid_src

            # Apply transformation to source
            src = (R_iter @ src.T).T + t_iter

            # Update cumulative transformation
            t = R_iter @ t + t_iter
            R = R_iter @ R

        return R, t, mean_error

    def derez_callback(self, msg: PointCloud2):
        """
        Callback for LiDAR point cloud data.

        Uses ICP to align consecutive scans and compute the transformation,
        which is used to correct odometry drift.

        Args:
            msg (PointCloud2): Point cloud from /derez_lidar_base/PointCloud2
        """

        # Parse the incoming point cloud
        new_points = self.parse_pointcloud2(msg)

        if len(new_points) == 0:
            self.get_logger().warn('Received empty point cloud', throttle_duration_sec=5.0)
            return

        # If this is the first scan, just store it
        if self.prev_lidar is None:
            self.prev_lidar = msg
            self.get_logger().info(f'Initialized ICP with {len(new_points)} points')
            return
        
        # Get raw odometry message that matches this LiDAR timestamp
        synced_odom_current = self.get_synced_odom(msg.header.stamp)
        synced_odom_previous = self.get_synced_odom(self.prev_lidar.header.stamp)

        if synced_odom_previous is None or synced_odom_current is None:
            self.get_logger().warn('No odometry data available for synchronization')
            return

        # You can now use synced_odom which has the timestamp closest to the LiDAR message
        # For example, extract position and orientation:
        odom_x_current = synced_odom_current.pose.pose.position.x
        odom_y_current = synced_odom_current.pose.pose.position.y
        q_current = synced_odom_current.pose.pose.orientation
        odom_yaw_current = np.arctan2(2*(q_current.w*q_current.z+q_current.x*q_current.y), 1-2*(q_current.y*q_current.y+q_current.z*q_current.z))

        odom_x_previous = synced_odom_previous.pose.pose.position.x
        odom_y_previous = synced_odom_previous.pose.pose.position.y
        q_previous = synced_odom_previous.pose.pose.orientation
        odom_yaw_previous = np.arctan2(2*(q_previous.w*q_previous.z+q_previous.x*q_previous.y), 1-2*(q_previous.y*q_previous.y+q_previous.z*q_previous.z))
        # Compute odometry-based transformation between previous and current
        self.delta_x_odom = odom_x_current - odom_x_previous
        self.delta_y_odom = odom_y_current - odom_y_previous
        self.delta_yaw_odom = odom_yaw_current - odom_yaw_previous



        # Apply odometry-based transformation to previous scan points
        # This provides motion compensation for better ICP initialization
        transformed_prev_points = self.forward_transform()


        # Perform ICP alignment
        # If motion compensation succeeded, use transformed points; otherwise use raw previous scan
        if transformed_prev_points is not None:
            R, t, error = self.icp(new_points, transformed_prev_points)
            # self.get_logger().debug('Using motion-compensated ICP', throttle_duration_sec=5.0)
            dx_local = t[0]
            dy_local = t[1]
            theta = np.arctan2(R[1, 0], R[0, 0])

            self.icp_correction_x = dx_local
            self.icp_correction_y = dy_local
            self.icp_correction_theta = theta
        else:
            self.icp_correction_x = 0.0
            self.icp_correction_y = 0.0
            self.icp_correction_theta = 0.0
            error = float('inf')
            self.get_logger().warn('Motion compensation failed, skipping ICP correction')
    
        # Update stored point cloud for next iteration
        self.prev_lidar = msg

        

    def forward_transform(self):
        '''
        Take the previous lidar data and apply the transform created by the change
        in the wheel odometry between the previous scan and the current one.

        This implements motion compensation: the previous scan is transformed
        according to the robot's odometry-based movement, providing a better
        initial alignment for ICP.

        Returns:
            np.ndarray: Transformed previous lidar points (Nx2 array)
        '''
        if self.prev_lidar is None:
            return None

        # Get the odometry deltas (computed in derez_callback)
        dx = -self.delta_x_odom
        dy = -self.delta_y_odom
        dtheta = -self.delta_yaw_odom

        # Create rotation matrix for the odometry rotation
        cos_theta = np.cos(dtheta)
        sin_theta = np.sin(dtheta)
        R_odom = np.array([[cos_theta, -sin_theta],
                           [sin_theta,  cos_theta]])

        # Translation vector
        t_odom = np.array([dx, dy])

        prev_points = self.parse_pointcloud2(self.prev_lidar)

        # Apply transformation to each point in the previous scan
        # Formula: p_new = R * p_old + t
        transformed_points = (R_odom @ prev_points.T).T + t_odom
        
        return transformed_points


    def update_Odometry(self, Odom):
        """
        Callback function that processes incoming odometry data,
        corrects it relative to the initial position/orientation,
        applies ICP-based corrections, and publishes the corrected odometry.
        """
        # Store raw odometry message in buffer for timestamp synchronization
        self.odom_buffer.append(Odom)

        position = Odom.pose.pose.position

        # Orientation uses the quaternion parametrization.
        # To get the angular position along the z-axis, the following equation is required.
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))

        if self.Init:
            # The initial data is stored to be subtracted from all the other values
            # as we want to start at position (0,0) and orientation 0
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],
                             [-np.sin(self.Init_ang), np.cos(self.Init_ang)]])
            self.Init_pos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Init_pos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Init_pos.z = position.z
            self.get_logger().info(f'Initial odometry set: pos=({self.Init_pos.x:.3f}, {self.Init_pos.y:.3f}), ang={self.Init_ang:.3f}')

        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],
                         [-np.sin(self.Init_ang), np.cos(self.Init_ang)]])

        # We subtract the initial values (wheel odometry correction)
        self.globalPos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos.x
        self.globalPos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos.y
        self.globalAng = orientation - self.Init_ang
        
        # Apply ICP-based correction

        # Publish corrected odometry
        corrected_odom = Odometry()
        corrected_odom.header = Odom.header
        corrected_odom.child_frame_id = 'base_link'
        corrected_odom.pose.pose.position.x = self.globalPos.x + self.icp_correction_x
        corrected_odom.pose.pose.position.y = self.globalPos.y + self.icp_correction_y
        corrected_odom.pose.pose.position.z = 0.0
        # Convert corrected orientation back to quaternion
        corrected_yaw = self.globalAng + self.icp_correction_theta
        qz = np.sin(corrected_yaw / 2.0)
        qw = np.cos(corrected_yaw / 2.0)
        corrected_odom.pose.pose.orientation.x = 0.0
        corrected_odom.pose.pose.orientation.y = 0.0
        corrected_odom.pose.pose.orientation.z = qz
        corrected_odom.pose.pose.orientation.w = qw
        self.odom_pub.publish(corrected_odom)

        self.get_logger().info(f'raw_odom=({self.globalPos.x:.3f}, {self.globalPos.y:.3f}, {self.globalAng:.3f}), icp_corr=({self.icp_correction_x:.3f}, {self.icp_correction_y:.3f}, {self.icp_correction_theta:.3f}), total_odom=({corrected_odom.pose.pose.position.x:.3f}, {corrected_odom.pose.pose.position.y:.3f}, {corrected_yaw:.3f})')


def main(args=None):
    """
    Main entry point for the Odometry Correction node.

    Initializes ROS2, creates the node instance, and spins until shutdown.
    Ensures proper cleanup on exit.

    Args:
        args: Command line arguments (optional)
    """
    rclpy.init(args=args)
    node = OdometryCorrectNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

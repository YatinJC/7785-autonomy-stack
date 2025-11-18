#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry
import math
import numpy as np
import time
import cv2
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

class AiDriver(Node):
    def __init__(self):
        super().__init__('ai_driver')
        
        # Parameters
        self.linear_speed = 0.15
        self.angular_speed = 0.5
        self.turn_tolerance = 0.1  # radians
        self.obstacle_dist = 0.5   # meters
        self.center_tolerance = 0.05 # Normalized image x tolerance
        
        # State
        self.state = 'DRIVING'  # DRIVING, CENTERING, CLASSIFYING, TURNING, STOPPED, FINISHED
        self.last_sign = 'empty'
        self.last_sign_time = 0
        self.sign_timeout = 1.0
        
        # Turning
        self.target_yaw = 0.0
        self.current_yaw = 0.0
        self.turn_direction = 0 
        
        # Lidar
        self.scan_data = None
        
        # Vision
        self.bridge = CvBridge()
        self.current_frame = None
        self.image_width = 0
        self.sign_centroid_x = None # Normalized -1 to 1
        
        # Classification
        self.classification_start_time = 0
        self.classification_duration = 2.0 
        self.detected_signs_buffer = []
        
        # Subs/Pubs
        self.create_subscription(String, '/detected_sign', self.sign_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(CompressedImage, '/image_raw/compressed', self.image_callback, 10)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Control Loop
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('AI Driver initialized with Color-Based Centering')

    def sign_callback(self, msg):
        self.last_sign = msg.data
        self.last_sign_time = time.time()
        if self.state == 'CLASSIFYING':
            self.detected_signs_buffer.append(msg.data)
        
    def scan_callback(self, msg):
        self.scan_data = msg
        
    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None: return
            
            self.current_frame = frame
            self.image_width = frame.shape[1]
            
            # Process for sign centroid (Red, Blue, or Green signs)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define color ranges (OpenCV HSV: H=0-179, S=0-255, V=0-255)
            
            # Blue (Left/Right) ~100-140
            lower_blue = np.array([100, 100, 50])
            upper_blue = np.array([140, 255, 255])
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Red (Stop/Do Not Enter) ~0-10 and ~170-180
            lower_red1 = np.array([0, 100, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 50])
            upper_red2 = np.array([180, 255, 255])
            mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), 
                                      cv2.inRange(hsv, lower_red2, upper_red2))
            
            # Green (Goal) ~40-80
            lower_green = np.array([40, 100, 50])
            upper_green = np.array([80, 255, 255])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            
            # Combine all masks
            mask = cv2.bitwise_or(mask_blue, mask_red)
            mask = cv2.bitwise_or(mask, mask_green)
            
            # Debug logging for color detection
            if self.state == 'CENTERING':
                blue_cnts, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                red_cnts, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                green_cnts, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                max_blue = max([cv2.contourArea(c) for c in blue_cnts]) if blue_cnts else 0
                max_red = max([cv2.contourArea(c) for c in red_cnts]) if red_cnts else 0
                max_green = max([cv2.contourArea(c) for c in green_cnts]) if green_cnts else 0
                
                if max_blue > 500 or max_red > 500 or max_green > 500:
                    colors = []
                    if max_blue > 500: colors.append(f"Blue({int(max_blue)})")
                    if max_red > 500: colors.append(f"Red({int(max_red)})")
                    if max_green > 500: colors.append(f"Green({int(max_green)})")
                    self.get_logger().info(f"Centering on: {', '.join(colors)}")
            
            # Morphological cleanup
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Largest blob is likely the sign
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 500: # Min area noise filter
                    M = cv2.moments(largest)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        # Normalize to -1.0 (left) to 1.0 (right)
                        self.sign_centroid_x = (2.0 * cx / self.image_width) - 1.0
                    else:
                        self.sign_centroid_x = None
                else:
                    self.sign_centroid_x = None
            else:
                self.sign_centroid_x = None
                
        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def start_turn(self, angle_deg):
        self.state = 'TURNING'
        angle_rad = math.radians(angle_deg)
        self.target_yaw = self.normalize_angle(self.current_yaw + angle_rad)
        self.get_logger().info(f'Starting turn: {angle_deg} deg')

    def control_loop(self):
        twist = Twist()
        
        # --- STATE MACHINE ---
        
        if self.state == 'FINISHED':
            self.cmd_pub.publish(twist)
            return

        # 1. DRIVING
        if self.state == 'DRIVING':
            # Check for obstacles
            if self.scan_data:
                # Check front cone
                ranges = np.array(self.scan_data.ranges)
                ranges[ranges == float('inf')] = 10.0
                n = len(ranges)
                window = int(n / 12) # +/- 30 deg
                front_ranges = np.concatenate((ranges[-window:], ranges[:window]))
                min_dist = np.min(front_ranges)
                
                if min_dist < self.obstacle_dist:
                    self.get_logger().warn(f'Wall detected ({min_dist:.2f}m). Switching to CENTERING.')
                    self.state = 'CENTERING'
                    twist.linear.x = 0.0
                    self.cmd_pub.publish(twist)
                    return
                
            # Drive forward
            twist.linear.x = self.linear_speed
            
        # 2. CENTERING (Visual Servoing)
        elif self.state == 'CENTERING':
            if self.sign_centroid_x is not None:
                err = self.sign_centroid_x
                
                if abs(err) < self.center_tolerance:
                    self.get_logger().info('Sign centered. Switching to CLASSIFYING.')
                    self.state = 'CLASSIFYING'
                    self.classification_start_time = time.time()
                    self.detected_signs_buffer = []
                    twist.angular.z = 0.0
                else:
                    # P-Controller for rotation
                    twist.angular.z = -1.5 * err 
                    twist.angular.z = max(min(twist.angular.z, self.angular_speed), -self.angular_speed)
            else:
                # Can't see sign? Rotate slowly to scan.
                twist.angular.z = 0.3
                
        # 3. CLASSIFYING
        elif self.state == 'CLASSIFYING':
            if time.time() - self.classification_start_time > self.classification_duration:
                # Process buffer
                if not self.detected_signs_buffer:
                    decision = 'empty'
                else:
                    from collections import Counter
                    counts = Counter(self.detected_signs_buffer)
                    decision = counts.most_common(1)[0][0]
                
                self.get_logger().info(f'Classification complete: {decision}')
                
                # React
                if decision == 'left':
                    self.start_turn(90)
                elif decision == 'right':
                    self.start_turn(-90)
                elif decision == 'do_not_enter':
                    self.start_turn(180)
                elif decision == 'stop':
                    self.state = 'STOPPED'
                elif decision == 'goal':
                    self.state = 'FINISHED'
                    self.get_logger().info('GOAL REACHED!')
                else:
                    self.get_logger().warn('No sign detected on wall. Turning left to explore.')
                    self.start_turn(90)
            else:
                # Waiting for data
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                
        # 4. TURNING (Open Loop / Odometry)
        elif self.state == 'TURNING':
            diff = self.normalize_angle(self.target_yaw - self.current_yaw)
            if abs(diff) < self.turn_tolerance:
                self.state = 'DRIVING'
                self.get_logger().info('Turn complete. Resuming drive.')
                twist.angular.z = 0.0
            else:
                twist.angular.z = 1.0 * diff
                twist.angular.z = max(min(twist.angular.z, self.angular_speed), -self.angular_speed)
                
        # 5. STOPPED
        elif self.state == 'STOPPED':
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = AiDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

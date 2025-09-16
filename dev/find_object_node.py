#!/usr/bin/env python3

# color_tracker_node_region.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
import cv2
from cv_bridge import CvBridge
import numpy as np

class ColorTrackerNode(Node):
    def __init__(self):
        super().__init__('color_tracker_node')
        
        # Parameters
        self.declare_parameter('tolerance_h', 10)
        self.declare_parameter('tolerance_s', 50)
        self.declare_parameter('tolerance_v', 50)
        
        self.selected_hsv = None
        self.bridge = CvBridge()
        
        # Subscribers and Publishers
        self._img_subscriber = self.create_subscription(CompressedImage, ’/image_raw/compressed’, self._image_callback, qos_profile)
        self.publisher = self.create_publisher(Point, '/color_centroid', 10)
        
        # Mouse callback
        cv2.namedWindow('Original')
        cv2.setMouseCallback('Original', self.mouse_callback)
        
        self.current_frame = None
        self.current_hsv = None
        self.mouse_x, self.mouse_y = -1, -1

        self.get_logger().info("Color Tracker Node Initialized")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current_hsv is not None:
            h, w = self.current_hsv.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                self.selected_hsv = self.current_hsv[y, x].copy()
                self.mouse_x, self.mouse_y = x, y
                self.get_logger().info(f"Selected HSV: {self.selected_hsv}")

    def create_hsv_mask(self, hsv_image):
        """Create mask based on HSV similarity to target pixel"""
        if self.selected_hsv is None:
            return np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        
        h_target = int(self.selected_hsv[0])
        s_target = int(self.selected_hsv[1])
        v_target = int(self.selected_hsv[2])

        tol_h = self.get_parameter('tolerance_h').value
        tol_s = self.get_parameter('tolerance_s').value
        tol_v = self.get_parameter('tolerance_v').value

        lower = np.array([max(0, h_target - tol_h),
                          max(0, s_target - tol_s),
                          max(0, v_target - tol_v)])
        upper = np.array([min(180, h_target + tol_h),
                          min(255, s_target + tol_s),
                          min(255, v_target + tol_v)])
        
        mask = cv2.inRange(hsv_image, lower, upper)
        
        # Optional: refine with morphology
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame = cv2.flip(frame, 1)
        
        self.current_frame = frame.copy()
        self.current_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        display_frame = frame.copy()
        
        mask = self.create_hsv_mask(self.current_hsv)

        centroid_msg = Point()
        centroid_msg.x = 0.0
        centroid_msg.y = 0.0
        centroid_msg.z = 0.0

        if self.selected_hsv is not None and self.mouse_x >= 0:
            # Use floodFill from clicked pixel
            mask_flood = mask.copy()
            h, w = mask.shape
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(mask_flood, flood_mask, (self.mouse_x, self.mouse_y), 255)
            
            # Extract all pixels filled
            ys, xs = np.where(flood_mask[1:-1, 1:-1] == 1)
            if len(xs) > 0:
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
                
                # Normalize to [-1, 1]
                centroid_msg.x = (cx / w) * 2 - 1
                centroid_msg.y = (cy / h) * 2 - 1
                
                # Draw centroid
                cv2.circle(display_frame, (cx, cy), 7, (255, 0, 0), -1)
                cv2.circle(display_frame, (cx, cy), 9, (255, 255, 255), 2)
            
            # Optional: show flood region
            result = cv2.bitwise_and(frame, frame, mask=mask_flood)
        else:
            result = np.zeros_like(frame)
        
        # Publish centroid
        self.publisher.publish(centroid_msg)
        
        # Show selected pixel
        if self.mouse_x >= 0 and self.mouse_y >= 0:
            cv2.circle(display_frame, (self.mouse_x, self.mouse_y), 5, (0, 255, 0), -1)
            cv2.circle(display_frame, (self.mouse_x, self.mouse_y), 7, (0, 0, 0), 1)
        
        cv2.imshow('Original', display_frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ColorTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

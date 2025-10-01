#!/usr/bin/env python3
# Yatin Chandar and Arun Showry Busani

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
import cv2
import numpy as np
from cv_bridge import CvBridge

class ColorDetectionNode(Node):
    def __init__(self):
        super().__init__('color_detection_node')

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # State variables
        self.selected_hsv = None
        self.tolerance_h = 10.0  # Hue tolerance
        self.tolerance_s = 50.0  # Saturation tolerance
        self.tolerance_v = 50.0  # Value tolerance
        self.current_frame = None
        self.current_hsv = None
        self.selected_pixel = None  # Store selected pixel coordinates
        self.image_width = None

        # Publishers
        self.coord_publisher = self.create_publisher(
            Float32,
            '/x_value/Float32',
            10
        )
        self.masked_publisher = self.create_publisher(
            CompressedImage,
            '/masked/compressed',
            10
        )
        self.size_publisher = self.create_publisher(
            Float32,
            '/obj_width/Float32',
            10
        )

        # Subscribers
        self.image_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            10
        )
        self.hue_subscriber = self.create_subscription(
            Float32,
            '/hue',
            self.hue_callback,
            10
        )
        self.sat_subscriber = self.create_subscription(
            Float32,
            '/sat',
            self.sat_callback,
            10
        )
        self.val_subscriber = self.create_subscription(
            Float32,
            '/val',
            self.val_callback,
            10
        )
        self.selection_subscriber = self.create_subscription(
            Point,
            '/selection',
            self.selection_callback,
            10
        )

        self.get_logger().info('Color Detection Node initialized')
        self.get_logger().info(f'Default tolerances - H: {self.tolerance_h}, S: {self.tolerance_s}, V: {self.tolerance_v}')

    def hue_callback(self, msg):
        self.tolerance_h = msg.data
        self.get_logger().debug(f'Hue tolerance updated to: {self.tolerance_h}')

    def sat_callback(self, msg):
        self.tolerance_s = msg.data
        self.get_logger().debug(f'Saturation tolerance updated to: {self.tolerance_s}')

    def val_callback(self, msg):
        self.tolerance_v = msg.data
        self.get_logger().debug(f'Value tolerance updated to: {self.tolerance_v}')

    def selection_callback(self, msg):
        """Handle pixel selection from viewer node"""
        self.selected_pixel = (int(msg.x), int(msg.y))

        # If we have a current HSV frame, calculate the HSV values for the selected pixel
        if self.current_hsv is not None:
            height, width = self.current_hsv.shape[:2]
            x, y = self.selected_pixel

            if 0 <= x < width and 0 <= y < height:
                self.selected_hsv = self.current_hsv[y, x].copy()
                self.get_logger().info(
                    f'Selected pixel at ({x}, {y}) - HSV: H={self.selected_hsv[0]}, '
                    f'S={self.selected_hsv[1]}, V={self.selected_hsv[2]}'
                )
            else:
                self.get_logger().warn(f'Selected pixel ({x}, {y}) is out of bounds')

    def create_hsv_mask(self, hsv_image, target_hsv, tol_h, tol_s, tol_v):
        """Create a mask based on HSV similarity to target color"""
        if target_hsv is None:
            return np.zeros(hsv_image.shape[:2], dtype=np.uint8)

        h_target = int(target_hsv[0])
        s_target = int(target_hsv[1])
        v_target = int(target_hsv[2])

        # Convert float tolerances to int
        tol_h = int(tol_h)
        tol_s = int(tol_s)
        tol_v = int(tol_v)

        # Define lower and upper bounds for each channel
        # Hue wraps around at 180 (in OpenCV HSV)
        if h_target - tol_h < 0:
            # Hue wraps around low end
            lower1 = np.array([0, max(0, s_target - tol_s), max(0, v_target - tol_v)])
            upper1 = np.array([h_target + tol_h, min(255, s_target + tol_s), min(255, v_target + tol_v)])
            lower2 = np.array([180 + (h_target - tol_h), max(0, s_target - tol_s), max(0, v_target - tol_v)])
            upper2 = np.array([180, min(255, s_target + tol_s), min(255, v_target + tol_v)])
            mask1 = cv2.inRange(hsv_image, lower1, upper1)
            mask2 = cv2.inRange(hsv_image, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        elif h_target + tol_h > 180:
            # Hue wraps around high end
            lower1 = np.array([h_target - tol_h, max(0, s_target - tol_s), max(0, v_target - tol_v)])
            upper1 = np.array([180, min(255, s_target + tol_s), min(255, v_target + tol_v)])
            lower2 = np.array([0, max(0, s_target - tol_s), max(0, v_target - tol_v)])
            upper2 = np.array([(h_target + tol_h) - 180, min(255, s_target + tol_s), min(255, v_target + tol_v)])
            mask1 = cv2.inRange(hsv_image, lower1, upper1)
            mask2 = cv2.inRange(hsv_image, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # No hue wraparound
            lower = np.array([h_target - tol_h, max(0, s_target - tol_s), max(0, v_target - tol_v)])
            upper = np.array([h_target + tol_h, min(255, s_target + tol_s), min(255, v_target + tol_v)])
            mask = cv2.inRange(hsv_image, lower, upper)

        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def image_callback(self, msg):
        """Process incoming compressed images"""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                self.get_logger().error('Failed to decode compressed image')
                return

            self.current_frame = frame
            self.image_width = frame.shape[1]
            # Convert to HSV
            self.current_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Create mask based on selected HSV value
            if self.selected_hsv is not None:
                mask = self.create_hsv_mask(
                    self.current_hsv,
                    self.selected_hsv,
                    self.tolerance_h,
                    self.tolerance_s,
                    self.tolerance_v
                )

                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # If contours found, find the largest one and publish its center
                if len(contours) > 0:
                    largest_contour = max(contours, key=cv2.contourArea)
                    approx_width = cv2.boundingRect(largest_contour)[2]
                    # Calculate centroid of largest contour
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # Publish normalized x coordinate
                        if self.image_width is not None:
                            norm_x = Float32()
                            norm_x.data = (2*cx/self.image_width) - 1
                            self.coord_publisher.publish(norm_x)
                            self.size_publisher.publish(Float32(data=approx_width/self.image_width))

                        self.get_logger().debug(f'Published center coordinate: ({cx}, {cy})')

                # Create masked image (apply mask to original frame)
                masked_image = cv2.bitwise_and(frame, frame, mask=mask)

            else:
                # No color selected yet - publish black mask
                mask = np.zeros(self.current_hsv.shape[:2], dtype=np.uint8)
                masked_image = np.zeros_like(frame)

            # Compress and publish masked image
            _, buffer = cv2.imencode('.jpg', masked_image)
            masked_msg = CompressedImage()
            masked_msg.header.stamp = self.get_clock().now().to_msg()
            masked_msg.format = "jpeg"
            masked_msg.data = buffer.tobytes()
            self.masked_publisher.publish(masked_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = ColorDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Color Detection Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


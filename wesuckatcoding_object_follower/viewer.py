#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
import cv2
import numpy as np
from cv_bridge import CvBridge
import threading

class ViewerControlNode(Node):
    def __init__(self):
        super().__init__('viewer_control_node')

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # State variables
        self.current_frame = None
        self.masked_frame = None
        self.display_frame = None
        self.mouse_x, self.mouse_y = -1, -1
        self.selected_pixel = None

        # Tolerance values (matching defaults from robot node)
        self.tolerance_h = 10.0
        self.tolerance_s = 50.0
        self.tolerance_v = 50.0

        # Thread lock for thread-safe image updates
        self.lock = threading.Lock()

        # Publishers for control messages
        self.selection_publisher = self.create_publisher(
            Point,
            '/selection',
            10
        )
        self.hue_publisher = self.create_publisher(
            Float32,
            '/hue',
            10
        )
        self.sat_publisher = self.create_publisher(
            Float32,
            '/sat',
            10
        )
        self.val_publisher = self.create_publisher(
            Float32,
            '/val',
            10
        )

        # Subscribers for image data
        self.image_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            10
        )
        self.masked_subscriber = self.create_subscription(
            CompressedImage,
            '/masked/compressed',
            self.masked_callback,
            10
        )

        # Create OpenCV windows
        cv2.namedWindow('Camera View')
        cv2.namedWindow('Masked View')
        cv2.namedWindow('Controls')

        # Set mouse callback for pixel selection
        cv2.setMouseCallback('Camera View', self.mouse_callback)

        # Create trackbars for tolerance adjustment
        cv2.createTrackbar('Hue Tolerance', 'Controls', int(self.tolerance_h), 90, self.on_hue_change)
        cv2.createTrackbar('Sat Tolerance', 'Controls', int(self.tolerance_s), 100, self.on_sat_change)
        cv2.createTrackbar('Val Tolerance', 'Controls', int(self.tolerance_v), 100, self.on_val_change)

        # Create a blank control panel image with instructions
        self.create_control_panel()

        # Start the display thread
        self.display_thread = threading.Thread(target=self.display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()

        self.get_logger().info('Viewer Control Node initialized')
        self.print_instructions()

    def print_instructions(self):
        """Print usage instructions to console"""
        self.get_logger().info('\n' + '='*40)
        self.get_logger().info('Color Detection Viewer - Instructions:')
        self.get_logger().info('='*40)
        self.get_logger().info('- Click on any pixel in "Camera View" to select a color')
        self.get_logger().info('- Adjust tolerance trackbars in "Controls" window')
        self.get_logger().info('- Green circle marks your selection point')
        self.get_logger().info('- Press "r" to reset selection')
        self.get_logger().info('- Press "q" to quit')
        self.get_logger().info('='*40 + '\n')

    def create_control_panel(self):
        """Create a control panel image with instructions"""
        panel = np.zeros((200, 400, 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)  # Dark gray background

        # Add text instructions
        instructions = [
            "Color Detection Controls",
            "",
            "Click in Camera View to select color",
            "Adjust sliders to change tolerance",
            "",
            "Keys:",
            "  'r' - Reset selection",
            "  'q' - Quit application"
        ]

        y_offset = 20
        for text in instructions:
            if text.startswith("Color Detection"):
                cv2.putText(panel, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif text.startswith("Keys:"):
                cv2.putText(panel, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(panel, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 20

        cv2.imshow('Controls', panel)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for pixel selection"""
        if event == cv2.EVENT_LBUTTONDOWN and self.current_frame is not None:
            with self.lock:
                height, width = self.current_frame.shape[:2]
                if 0 <= x < width and 0 <= y < height:
                    self.mouse_x, self.mouse_y = x, y
                    self.selected_pixel = (x, y)

                    # Publish selection to robot node
                    selection_msg = Point()
                    selection_msg.x = float(x)
                    selection_msg.y = float(y)
                    selection_msg.z = 0.0  # Not used, but Point requires it
                    self.selection_publisher.publish(selection_msg)

                    self.get_logger().info(f'Selected pixel at ({x}, {y})')

    def on_hue_change(self, value):
        """Callback for hue tolerance trackbar"""
        self.tolerance_h = float(value)
        msg = Float32()
        msg.data = self.tolerance_h
        self.hue_publisher.publish(msg)
        self.get_logger().debug(f'Hue tolerance changed to: {value}')

    def on_sat_change(self, value):
        """Callback for saturation tolerance trackbar"""
        self.tolerance_s = float(value)
        msg = Float32()
        msg.data = self.tolerance_s
        self.sat_publisher.publish(msg)
        self.get_logger().debug(f'Saturation tolerance changed to: {value}')

    def on_val_change(self, value):
        """Callback for value tolerance trackbar"""
        self.tolerance_v = float(value)
        msg = Float32()
        msg.data = self.tolerance_v
        self.val_publisher.publish(msg)
        self.get_logger().debug(f'Value tolerance changed to: {value}')

    def image_callback(self, msg):
        """Handle incoming camera images"""
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                with self.lock:
                    self.current_frame = frame.copy()
                    self.update_display_frame()

        except Exception as e:
            self.get_logger().error(f'Error processing camera image: {str(e)}')

    def masked_callback(self, msg):
        """Handle incoming masked images"""
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                with self.lock:
                    self.masked_frame = frame

        except Exception as e:
            self.get_logger().error(f'Error processing masked image: {str(e)}')

    def update_display_frame(self):
        """Update the display frame with overlays"""
        if self.current_frame is None:
            return

        self.display_frame = self.current_frame.copy()

        # Draw selection marker if a pixel is selected
        if self.mouse_x >= 0 and self.mouse_y >= 0:
            cv2.circle(self.display_frame, (self.mouse_x, self.mouse_y), 5, (0, 255, 0), -1)
            cv2.circle(self.display_frame, (self.mouse_x, self.mouse_y), 7, (0, 0, 0), 1)

            # Add text overlay
            cv2.putText(self.display_frame, f"Selected: ({self.mouse_x}, {self.mouse_y})",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(self.display_frame, "Click to select color",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display current tolerance values
        cv2.putText(self.display_frame,
                   f"Tolerances - H: {int(self.tolerance_h)} S: {int(self.tolerance_s)} V: {int(self.tolerance_v)}",
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def display_loop(self):
        """Main display loop running in separate thread"""
        while rclpy.ok():
            with self.lock:
                # Display camera view with overlays
                if self.display_frame is not None:
                    cv2.imshow('Camera View', self.display_frame)

                # Display masked view
                if self.masked_frame is not None:
                    cv2.imshow('Masked View', self.masked_frame)
                elif self.current_frame is not None:
                    # Show blank frame if no mask received yet
                    blank = np.zeros_like(self.current_frame)
                    cv2.putText(blank, "Waiting for masked image...",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                    cv2.imshow('Masked View', blank)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('Quit requested')
                rclpy.shutdown()
                break
            elif key == ord('r'):
                # Reset selection
                with self.lock:
                    self.mouse_x, self.mouse_y = -1, -1
                    self.selected_pixel = None
                    self.update_display_frame()

                # Send reset signal (pixel at -1, -1)
                selection_msg = Point()
                selection_msg.x = -1.0
                selection_msg.y = -1.0
                selection_msg.z = 0.0
                self.selection_publisher.publish(selection_msg)

                self.get_logger().info('Selection reset')

        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = ViewerControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Viewer Control Node')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

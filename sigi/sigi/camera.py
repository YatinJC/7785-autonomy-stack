#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import torch
import timm
import numpy as np
from PIL import Image as PILImage
from torchvision import transforms
import os
from sigi_interfaces.srv import ReadSign

class CameraProcessor(Node):
    def __init__(self):
        super().__init__('camera_processor')

        # Parameters
        # UPDATE: Point this to your new model file name
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(script_dir, 'mobilenetv3_shape_classifier.pth')
        self.declare_parameter('model_path', default_model_path)
        #self.declare_parameter('camera_topic', '/simulated_camera/image_raw') # sim camera
        self.declare_parameter('camera_topic', '/image_raw/compressed') # real camera
        self.declare_parameter('use_compressed', True)
        
        model_path = self.get_parameter('model_path').value
        camera_topic = self.get_parameter('camera_topic').value
        use_compressed = self.get_parameter('use_compressed').value
        
        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Store latest frame for on-demand processing
        self.latest_frame = None
        self.frame_lock = False  # Simple flag to prevent race conditions

        # Load Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        self.classes = ['empty', 'left', 'right', 'do_not_enter', 'stop', 'goal']
        self.model = self._load_model(model_path)
        
        # Transforms (MUST MATCH TRAINING)
        # 1. Resize to 224x224
        # 2. ToTensor (converts 0-255 to 0.0-1.0)
        # 3. Normalize 1-channel input
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])
        
        # Subscribers
        if use_compressed:
            self.create_subscription(
                CompressedImage,
                camera_topic,
                self.compressed_image_callback,
                10
            )
        else:
            self.create_subscription(
                Image,
                camera_topic,
                self.image_callback,
                10
            )

        # Service for on-demand sign reading
        self.read_sign_service = self.create_service(
            ReadSign,
            '/read_sign',
            self.read_sign_callback
        )

        # Publisher for visualization (optional, for debugging)
        self.processed_feed_pub = self.create_publisher(CompressedImage, '/processed_feed/compressed', 10)

        self.get_logger().info('Camera processor initialized (MobileNetV3 + Canny) - On-demand mode')

    def _load_model(self, model_path):
        try:
            # Create MobileNetV3-Small with 1 input channel (for edges)
            # This matches the 'mobilenetv3_small_100' from your training script
            model = timm.create_model('mobilenetv3_small_100',
                                    pretrained=False,
                                    num_classes=6,
                                    in_chans=1) # CRITICAL: 1 channel input
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle state dict loading
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                    
                self.get_logger().info(f'Loaded model from {model_path}')
            else:
                self.get_logger().warn(f'Model file not found at {model_path}! Predictions will be random.')
                
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            return None

    def image_callback(self, msg):
        """Store latest frame for on-demand processing."""
        try:
            if not self.frame_lock:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.latest_frame = cv_image
        except Exception as e:
            self.get_logger().error(f'Error storing image: {e}')

    def compressed_image_callback(self, msg):
        """Store latest frame for on-demand processing."""
        try:
            if not self.frame_lock:
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                self.latest_frame = cv_image
        except Exception as e:
            self.get_logger().error(f'Error storing compressed image: {e}')

    def read_sign_callback(self, request, response):
        """Service callback for on-demand sign reading."""
        try:
            # Check if we have a frame
            if self.latest_frame is None:
                self.get_logger().warn('No frame available for sign reading')
                response.success = False
                response.class_name = 'empty'
                response.confidence = 0.0
                return response

            # Check if model is loaded
            if self.model is None:
                self.get_logger().error('Model not loaded')
                response.success = False
                response.class_name = 'empty'
                response.confidence = 0.0
                return response

            # Lock frame and process
            self.frame_lock = True
            cv_image = self.latest_frame.copy()
            self.frame_lock = False

            # Perform inference
            class_name, confidence, edges, cv_image_for_viz = self._process_image(cv_image)

            # Publish visualization (optional)
            self._publish_visualization(cv_image_for_viz, edges, class_name, confidence)

            # Fill response
            response.success = True
            response.class_name = class_name
            response.confidence = confidence

            self.get_logger().info(f'Sign reading: {class_name} ({confidence:.2f})')
            return response

        except Exception as e:
            self.get_logger().error(f'Error in sign reading service: {e}')
            response.success = False
            response.class_name = 'empty'
            response.confidence = 0.0
            return response

    def _process_image(self, cv_image):
        """Process image and return classification results."""
        # 1. Convert to HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # 2. Extract Saturation Channel (Index 1)
        # This separates "colorful things" (signs) from "dull things" (cardboard)
        saturation_channel = hsv[:, :, 1]

        # 3. Apply Canny Edge Detection to Saturation
        edges = cv2.Canny(saturation_channel, 100, 200)

        # 4. Convert to PIL and normalize
        pil_image = PILImage.fromarray(edges)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, predicted = torch.max(probs, 1)

            class_idx = predicted.item()
            class_name = self.classes[class_idx]
            confidence = conf.item()

        return class_name, confidence, edges, cv_image

    def _publish_visualization(self, cv_image, edges, class_name, confidence):
        """Publish visualization for debugging."""
        try:
            # Visualization: Show the Saturation Edge Map
            edge_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            combined_view = np.hstack((cv_image, edge_display))
            combined_view = cv2.resize(combined_view, (0,0), fx=0.5, fy=0.5)

            text = f"Pred: {class_name} ({confidence:.2f})"
            cv2.putText(combined_view, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Publish compressed image
            _, encoded_img = cv2.imencode('.jpg', combined_view)
            compressed_msg = CompressedImage()
            compressed_msg.header.stamp = self.get_clock().now().to_msg()
            compressed_msg.format = "jpeg"
            compressed_msg.data = encoded_img.tobytes()
            self.processed_feed_pub.publish(compressed_msg)
        except Exception as e:
            self.get_logger().warn(f'Error publishing visualization: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = CameraProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
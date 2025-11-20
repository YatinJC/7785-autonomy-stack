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

class CameraProcessor(Node):
    def __init__(self):
        super().__init__('camera_processor')
        
        # Parameters
        self.declare_parameter('model_path', '/home/yc/turtlebot3_ws/src/7785-autonomy-stack/splatch/mobilenetv4_sign_classifier.pth')
        self.declare_parameter('camera_topic', '/simulated_camera/image_raw')
        self.declare_parameter('use_compressed', False)
        
        model_path = self.get_parameter('model_path').value
        camera_topic = self.get_parameter('camera_topic').value
        use_compressed = self.get_parameter('use_compressed').value
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Load Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')
        
        self.classes = ['empty', 'left', 'right', 'do_not_enter', 'stop', 'goal']
        self.model = self._load_model(model_path)
        
        # Transforms (must match training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
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
            
        # Publishers
        self.sign_pub = self.create_publisher(String, '/detected_sign', 10)
        
        self.get_logger().info('Camera processor initialized')

    def _load_model(self, model_path):
        try:
            # Try creating MobileNetV4
            try:
                model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k',
                                        pretrained=False,
                                        num_classes=6)
            except:
                # Fallback to other variants if needed
                model = timm.create_model('mobilenetv3_large_100',
                                        pretrained=False,
                                        num_classes=6)
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                # Handle if state dict is nested
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
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self._process_image(cv_image)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def compressed_image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self._process_image(cv_image)
        except Exception as e:
            self.get_logger().error(f'Error processing compressed image: {e}')

    def _process_image(self, cv_image):
        if self.model is None:
            return
            
        # Preprocess
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, predicted = torch.max(probs, 1)
            
            class_idx = predicted.item()
            class_name = self.classes[class_idx]
            confidence = conf.item()
            
        # Publish result
        msg = String()
        msg.data = class_name
        self.sign_pub.publish(msg)
        
        # Visualization
        display_img = cv_image.copy()
        text = f"{class_name} ({confidence:.2f})"
        cv2.putText(display_img, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Camera View", display_img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CameraProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

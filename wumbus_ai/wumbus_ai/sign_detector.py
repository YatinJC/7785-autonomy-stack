#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
import torch
import timm
from torchvision import transforms
from PIL import Image
import os
from ament_index_python.packages import get_package_share_directory

class SignDetector(Node):
    def __init__(self):
        super().__init__('sign_detector')
        
        # Parameters
        self.declare_parameter('model_name', 'mobilenetv4_sign_classifier.pth')
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_name = self.get_parameter('model_name').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.device = self.get_parameter('device').value
        
        # Classes (must match training)
        self.classes = ['empty', 'left', 'right', 'do_not_enter', 'stop', 'goal']
        
        # Load Model
        self.load_model()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Subscribers & Publishers
        self.sub = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            10
        )
        self.pub = self.create_publisher(String, '/detected_sign', 10)
        
        self.get_logger().info(f'Sign Detector initialized on {self.device}')

    def load_model(self):
        try:
            pkg_share = get_package_share_directory('wumbus_ai')
            model_path = os.path.join(pkg_share, 'models', self.model_name)
            
            self.get_logger().info(f'Loading model from: {model_path}')
            
            # Create model architecture (must match training script)
            # Try to load MobileNetV4, fallback logic similar to training script
            try:
                self.model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k',
                                          pretrained=False,
                                          num_classes=len(self.classes))
            except:
                try:
                    self.model = timm.create_model('mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k',
                                              pretrained=False,
                                              num_classes=len(self.classes))
                except:
                    self.model = timm.create_model('mobilenetv3_large_100',
                                              pretrained=False,
                                              num_classes=len(self.classes))
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            self.get_logger().info('Model loaded successfully')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise e

    def image_callback(self, msg):
        try:
            # Decompress
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                return

            # Convert to PIL (RGB)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)
            
            # Preprocess
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            conf_val = confidence.item()
            class_idx = predicted.item()
            class_name = self.classes[class_idx]
            
            # Log detection
            if conf_val > self.conf_threshold:
                msg = String()
                msg.data = class_name
                self.pub.publish(msg)
                self.get_logger().info(f'Detected: {class_name} ({conf_val:.2f})')
            else:
                # Publish empty if low confidence? Or just nothing?
                # Let's publish "empty" if nothing is confident enough, or just silent.
                # Ideally, we want to be robust. If low confidence, maybe it's "empty".
                if class_name != 'empty':
                     self.get_logger().debug(f'Ignored low confidence {class_name}: {conf_val:.2f}')
            
        except Exception as e:
            self.get_logger().error(f'Error in inference: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = SignDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

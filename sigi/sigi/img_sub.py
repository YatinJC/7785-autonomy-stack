# Bare Bones Code to View the Image Published from the Turtlebot3 on a Remote Computer
# Intro to Robotics Research 7785
# Georgia Institute of Technology
# Sean Wilson, 2022

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

import sys

import numpy as np
import cv2

class MinimalVideoSubscriber(Node):

	def __init__(self):		
		# Creates the node.
		super().__init__('minimal_video_subscriber')

		# Set Parameters
		self.declare_parameter('show_image_bool', True)
		self.declare_parameter('window_name', "Processed Feed")

		#Determine Window Showing Based on Input
		self._display_image = bool(self.get_parameter('show_image_bool').value)

		# Declare some variables
		self._titleOriginal = self.get_parameter('window_name').value # Image Window Title	
		if(self._display_image):
		# Set Up Image Viewing
			cv2.namedWindow(self._titleOriginal, cv2.WINDOW_AUTOSIZE ) # Viewing Window
			cv2.moveWindow(self._titleOriginal, 50, 50) # Viewing Window Original Location
	
		#Set up QoS Profiles for passing images over WiFi
		image_qos_profile = QoSProfile(depth=5)
		image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
		image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE 
		image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT 

		#Declare that the minimal_video_subscriber node is subcribing to the /camera/image/compressed topic.
		self._video_subscriber = self.create_subscription(
				CompressedImage,
				'/processed_feed/compressed',
				self._image_callback,
				image_qos_profile)
		self._video_subscriber # Prevents unused variable warning.

	def _image_callback(self, CompressedImage):	
		# The "CompressedImage" is transformed to a color image in BGR space and is store in "_imgBGR"
		# Convert compressed image data to numpy array
		np_arr = np.frombuffer(CompressedImage.data, np.uint8)
		# Decode image using OpenCV
		self._imgBGR = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		if(self._display_image):
			# Display the image in a window
			self.show_image(self._imgBGR)
				

	def get_image(self):
		return self._imgBGR

	def get_user_input(self):
		return self._user_input

	def show_image(self, img):
		cv2.imshow(self._titleOriginal, img)
		# Cause a slight delay so image is displayed
		self._user_input=cv2.waitKey(50) #Use OpenCV keystroke grabber for delay.Done
		if self.get_user_input() == ord('q'):
				cv2.destroyAllWindows()
				raise SystemExit


def main():
	rclpy.init() #init routine needed for ROS2.
	video_subscriber = MinimalVideoSubscriber() #Create class object to be used.
	
	try:
		rclpy.spin(video_subscriber) # Trigger callback processing.		
	except SystemExit:
		rclpy.logging.get_logger("Camera Viewer Node Info...").info("Shutting Down")
	#Clean up and shutdown.
	video_subscriber.destroy_node()  
	rclpy.shutdown()


if __name__ == '__main__':
	main()

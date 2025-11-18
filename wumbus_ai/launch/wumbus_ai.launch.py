import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('wumbus_ai')
    
    return LaunchDescription([
        Node(
            package='wumbus_ai',
            executable='sign_detector',
            name='sign_detector',
            output='screen',
            parameters=[
                {'model_name': 'mobilenetv4_sign_classifier.pth'},
                {'confidence_threshold': 0.7}
            ]
        ),
        Node(
            package='wumbus_ai',
            executable='ai_driver',
            name='ai_driver',
            output='screen',
            parameters=[
                {'linear_speed': 0.15},
                {'angular_speed': 0.5}
            ]
        )
    ])

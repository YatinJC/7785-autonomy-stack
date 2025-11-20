from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
    # Arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false', # CHANGED: Must be false for real robot
        description='Use simulation (Gazebo) clock if true'
    )
    
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        # CHANGED: Verify this matches your specific camera driver output
        default_value='/image_raw/compressed', 
        description='Camera topic to subscribe to'
    )
    
    use_compressed_arg = DeclareLaunchArgument(
        'use_compressed',
        default_value='true', # CHANGED: Recommended for Wi-Fi
        description='Use compressed image transport'
    )
    
    # Nodes
    lidar_node = Node(
        package='sigi',
        executable='lidar_processor',
        name='lidar_processor',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )
    
    camera_node = Node(
        package='sigi',
        executable='camera',
        name='camera_processor',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'camera_topic': LaunchConfiguration('camera_topic'),
            'use_compressed': LaunchConfiguration('use_compressed'),
            # Default model path is hardcoded in node but can be overridden here if needed
        }]
    )
    
    controller_node = Node(
        package='sigi',
        executable='controller',
        name='cell_center_controller',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )
    
    # Event handler to start controller only after lidar processor has started
    controller_after_lidar = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=lidar_node,
            on_start=[controller_node]
        )
    )

    return LaunchDescription([
        use_sim_time_arg,
        camera_topic_arg,
        use_compressed_arg,
        lidar_node,
        camera_node,
        controller_after_lidar
    ])

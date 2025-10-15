

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare launch arguments
    resolution_arg = DeclareLaunchArgument(
        'resolution',
        default_value='0.5',
        description='Resolution for A* search grid (in meters)'
    )

    # Get launch configuration
    resolution = LaunchConfiguration('resolution')

    return LaunchDescription([
        resolution_arg,

        # Derez LiDAR node - processes laser scan data and publishes obstacle points
        Node(
            package='wumbus',
            executable='derez_lidar',
            name='derez_lidar',
            output='screen',
            parameters=[],
        ),

        # Goal tracker node - manages navigation goals and publishes current goal
        Node(
            package='wumbus',
            executable='goal_tracker',
            name='goal_tracker',
            output='screen',
            parameters=[],
        ),

        # Search node - performs A* path planning to navigate around obstacles
        Node(
            package='wumbus',
            executable='search_node',
            name='search_node',
            output='screen',
            parameters=[
                {'resolution': resolution}
            ],
        ),

        # Controller node - executes path by sending velocity commands
        Node(
            package='wumbus',
            executable='controller',
            name='controller',
            output='screen',
            parameters=[],
        ),
    ])

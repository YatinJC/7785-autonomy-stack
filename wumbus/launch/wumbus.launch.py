#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch import conditions


def generate_launch_description():
    # Get the path to the RViz config file
    rviz_config_file = PathJoinSubstitution([
        FindPackageShare('wumbus'),
        'config',
        'wumbus.rviz'
    ])

    # Launch argument for enabling/disabling RViz
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz for visualization'
    )

    return LaunchDescription([
        use_rviz_arg,

        # Odometry correction node - corrects raw odometry using wheel encoders + ICP
        # Subscribes to: /odom, /derez_lidar_base/PointCloud2
        # Publishes to: /odom_corrected
        Node(
            package='wumbus',
            executable='odom_correct',
            name='odom_correct',
            output='screen',
            parameters=[],
        ),

        # Derez LiDAR node - transforms laser scans to base_link frame
        # Subscribes to: /scan, /odom_corrected
        # Publishes to: /derez_lidar_base/PointCloud2
        Node(
            package='wumbus',
            executable='derez_lidar',
            name='derez_lidar',
            output='screen',
            parameters=[],
        ),

        # Transform to Global node - transforms LiDAR points to global frame
        # Subscribes to: /odom_corrected, /derez_lidar_base/PointCloud2
        # Publishes to: /obstacle_points/PointCloud2
        Node(
            package='wumbus',
            executable='transform_to_global',
            name='transform_to_global',
            output='screen',
            parameters=[],
        ),

        # Goal tracker node - manages navigation goals and publishes current goal
        # Subscribes to: user input or predefined goals
        # Publishes to: /current_goal/Point
        Node(
            package='wumbus',
            executable='goal_tracker',
            name='goal_tracker',
            output='screen',
            parameters=[],
        ),

        # Search node - performs A* path planning to navigate around obstacles
        # Subscribes to: /odom_corrected, /obstacle_points/PointCloud2, /current_goal/Point
        # Publishes to: /next_location/point
        Node(
            package='wumbus',
            executable='search_node',
            name='search_node',
            output='screen',
            parameters=[],
        ),

        # Controller node - executes path by sending velocity commands
        # Subscribes to: /odom_corrected, /next_location/point
        # Publishes to: /cmd_vel
        Node(
            package='wumbus',
            executable='controller',
            name='controller',
            output='screen',
            parameters=[],
        ),

        # Mapper node - builds occupancy grid map from LiDAR observations
        # Subscribes to: /obstacle_points/PointCloud2, /odom_corrected
        # Publishes to: /map
        Node(
            package='wumbus',
            executable='mapper',
            name='mapper',
            output='screen',
            parameters=[],
        ),

        # RViz node - visualization
        # Displays: map, goals, path, LiDAR, robot model
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file],
            condition=conditions.IfCondition(LaunchConfiguration('use_rviz'))
        ),
    ])

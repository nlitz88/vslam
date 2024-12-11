import os

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # Create launch configurations. Declare their corresponding launch arguments
    # within the LaunchDescription.
    left_camera_parameters_filepath = LaunchConfiguration('left_camera_parameters_filepath'),
    vicon_parameters_filepath = LaunchConfiguration('vicon_parameters_filepath')
    eurocmav_rosbag_filepath = LaunchConfiguration('eurocmav_rosbag_filepath')

    return launch.LaunchDescription([
        DeclareLaunchArgument(
            'left_camera_parameters_filepath',
            # default_value=os.path.join(get_package_share_directory('eurocmav_wrapper'), 'config', ''),
            description='Full path to the left camera calibration file as provided in the EuRoC MAV Dataset ASL files.'
        ),
        DeclareLaunchArgument(
            'vicon_parameters_filepath',
            # default_value=os.path.join(get_package_share_directory('eurocmav_wrapper'), 'config', ''),
            description='Full path to the vicon sensor parameter file from the EuRoC MAV Dataset ASL files.'
        ),
        DeclareLaunchArgument(
            'eurocmav_rosbag_filepath',
            # default_value=os.path.join(get_package_share_directory('eurocmav_wrapper'), 'config', ''),
            description='Full path to the EuRoC MAV dataset ROS 2 Bag directory.'
        ),
        # TODO: Add node that publishes each of the camera parameters.
        
        # TODO: Bring up the ROS bag and begin playing it.
    ])
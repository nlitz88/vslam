import os

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# NOTE: This launch file is only dedicated to bringing up / working with the
# EuRoC datasets WITH VICON DATA -- the datasets with only Leica position aren't
# integrated here. Could make a separate launch file for those later.

# Just as a convenient reference, one of the ORBSLAM2 ROS wrappers has a good
# launch file to follow:
# https://github.com/appliedAI-Initiative/orb_slam_2_ros/blob/ros2/ros/launch/orb_slam2_t265_stereo_launch.py
# Not specific to our application, but demos how to use launch arguments in a
# way that isn't super confusing.

def generate_launch_description():

    # General pattern for using "LaunchConfiguration" and
    # "DeclareLaunchArgument":
    # A LaunchConfiguration variable is basically just a variable that
    # allows us to access / reference / pass around a LaunchArgument that we
    # might declare later. If you don't create a LaunchConfiguration, then
    # you have no way of referencing (with a python variable) that Launch
    # Argument, and would have no way to pass its values to any nodes /
    # other Launch Descriptions.

    # Create launch configurations. Declare their corresponding launch arguments
    # within the LaunchDescription.
    left_camera_parameters_filepath = LaunchConfiguration('left_camera_parameters_filepath'),
    vicon_parameters_filepath = LaunchConfiguration('vicon_parameters_filepath')

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
        Node(
            package='vslam',
            executable='vo_node',
            name='vo_node'
        ),
        # TODO: Either launch a custom stereo node to compute the disparity or
        # use the stereo_image_proc system.
        Node(
            package='vslam',
            executable='stereo_node',
            name='stereo_node'),
        # Include the launch file from the eurocmav wrapper package. Reference
        # for doing all this below. Note that (for consistency) I opted to not
        # use their PythonLaunchDescriptionSource and instead just provided a
        # path.
        # https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Using-Substitutions.html#parent-launch-file
        launch.actions.IncludeLaunchDescription(
            launch_description_source=os.path.join(get_package_share_directory('eurocmav_wrapper'), 'launch', 'eurocmav.launch.py'),
            launch_arguments=[
                ('left_camera_parameters_filepath', left_camera_parameters_filepath),
                ('vicon_parameters_filepath', vicon_parameters_filepath)
            ]
        )
    ])
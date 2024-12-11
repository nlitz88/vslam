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

    # ACTUALLY, updated note on this: If you don't want to define these
    # variables to refere to these LaunchConfiguration values, you can literally
    # just use that same syntax to access the LaunchArgument values. I.e., when
    # passing in parameters, just use
    # LaunchConfiguration('left_camera_parameters_filepath'), for example! See
    # this page for more:
    # https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Using-ROS2-Launch-For-Large-Projects.html#setting-parameters-in-the-launch-file
    
    # Create launch configurations. Declare their corresponding launch arguments
    # within the LaunchDescription.
    vo_node_parameters_filepath = LaunchConfiguration
    left_camera_parameters_filepath = LaunchConfiguration('left_camera_parameters_filepath'),
    vicon_parameters_filepath = LaunchConfiguration('vicon_parameters_filepath')
    eurocmav_rosbag_filepath = LaunchConfiguration('eurocmav_rosbag_filepath')

    return launch.LaunchDescription([
        DeclareLaunchArgument(
            'vo_node_parameters_filepath',
            default_value=os.path.join(get_package_share_directory('vslam_bringup'), 'config', 'vo_params.yaml'),
            description='Path to parameter file for configuring the visual odometry pipeline.'
        ),
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
        Node(
            package='vslam',
            executable='vo_node',
            name='vo_node',
            remappings=[('left_image', 'camera/infra1/image_rect_raw'),
                        ('depth_image', 'camera/depth/image_rect_raw'),
                        ('camera_info', 'camera/infra1/camera_info')],
            parameters=[vo_node_parameters_filepath],
            # arguments=["--ros-args", "--log-level", "debug"]
        ),
        # TODO: Either launch a custom stereo node to compute the disparity or
        # use the stereo_image_proc system.
        # Node(
        #     package='vslam',
        #     executable='stereo_node',
        #     name='stereo_node'),
        # Include the launch file from the eurocmav wrapper package. Reference
        # for doing all this below. Note that (for consistency) I opted to not
        # use their PythonLaunchDescriptionSource and instead just provided a
        # path.
        # https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Using-Substitutions.html#parent-launch-file
        launch.actions.IncludeLaunchDescription(
            launch_description_source=os.path.join(get_package_share_directory('eurocmav_wrapper'), 'launch', 'eurocmav.launch.py'),
            launch_arguments=[
                ('left_camera_parameters_filepath', left_camera_parameters_filepath),
                ('vicon_parameters_filepath', vicon_parameters_filepath),
                ('eurocmav_rosbag_filepath', eurocmav_rosbag_filepath)
            ]
        )
    ])
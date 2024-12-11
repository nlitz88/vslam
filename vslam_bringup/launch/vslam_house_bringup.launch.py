import os
import launch
import launch_ros.actions

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    log_level = LaunchConfiguration("log_level")
    vo_params = LaunchConfiguration("vo_params")
    rqt_feature_matching_perspective = LaunchConfiguration("rqt_feature_matching_perspective")
    vo_debugging_rviz_config = LaunchConfiguration("vo_debugging_rviz_config")
    
    return launch.LaunchDescription([
        DeclareLaunchArgument(
            "log_level",
            default_value="info",
            description="The ROS logging level to use."
        ),
        DeclareLaunchArgument(
            "vo_params",
            default_value=os.path.join(get_package_share_directory("vslam_bringup"), "config", "vo_params.yaml"),
            description="The path to the VO parameters file."
        ),
        DeclareLaunchArgument(
            "rqt_feature_matching_perspective",
            default_value=os.path.join(get_package_share_directory("vslam_bringup"), "config", "feature-detection-matching.perspective"),
            description="The path to the RQT perspective file for feature detection and matching."
        ),
        DeclareLaunchArgument(
            "vo_debugging_rviz_config",
            default_value=os.path.join(get_package_share_directory("vslam_bringup"), "config", "vo_debugging.rviz"),
            description="The path to the RVIZ configuration file for VO debugging."
        ),
        launch_ros.actions.Node(
            package='vslam',
            executable='vo_node',
            name='vo_node',
            remappings=[('left_image', 'camera/infra1/image_rect_raw'),
                        ('depth_image', 'camera/depth/image_rect_raw'),
                        ('camera_info', 'camera/infra1/camera_info')],
            parameters=[vo_params],
            arguments=["--ros-args", "--log-level", log_level]
        ),
        # TODO: Bring up ros2 bag with the provided path. This path
        # should be a required argument to use the launch file. Would also be
        # nice to have arguments for the speed that the bag is playing at. Also,
        # I may not include this in the launch file if it means I can't play and
        # pause--would have to do that from rqt I think.

        # Bring up the rqt window with the feature detection and matching
        # visualizations and inlier plots using the saved config.
        launch_ros.actions.Node(
            package='rqt_gui',
            executable='rqt_gui',
            name='rqt',
            arguments=['--perspective-file', rqt_feature_matching_perspective]
        ),

        # Bring up RVIZ window with the provided debugging config.
        launch_ros.actions.Node(
            package='rviz2',
            executable='rviz2',
            name='vo_debugging_rviz',
            arguments=['-d', vo_debugging_rviz_config]
        ),

  ])
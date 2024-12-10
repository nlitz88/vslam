import os
import launch
import launch_ros.actions
import ament_index_python

def generate_launch_description():

    rqt_feature_matching_perspective = os.path.join(
        ament_index_python.packages.get_package_share_directory("vslam_bringup"),
        "config",
        "feature-detection-matching.perspective",
    )

    vo_debugging_rviz_config = os.path.join(
        ament_index_python.packages.get_package_share_directory("vslam_bringup"),
        "config",
        "vo_debugging.rviz",
    )

    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='vslam',
            executable='vo_node',
            name='vo_node',
            remappings=[('left_image', 'camera/infra1/image_rect_raw'),
                        ('depth_image', 'camera/depth/image_rect_raw'),
                        ('camera_info', 'camera/infra1/camera_info')]
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
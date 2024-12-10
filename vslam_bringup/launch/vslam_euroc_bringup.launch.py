import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([

        # TODO: Create a launch argument == the path to a parameter file. If one
        # is not provided a default file will be used == the config file in this
        # bringup package. This will allow us to reuse the same launch file but
        # with a different configuration down the road.
        
        launch_ros.actions.Node(
            package='vslam',
            executable='vo_node',
            name='vo_node'),
        launch_ros.actions.Node(
            package='vslam',
            executable='stereo_node',
            name='stereo_node'),
  ])
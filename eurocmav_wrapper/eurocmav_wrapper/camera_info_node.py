"""Node for parsing the provided EuRoC MAV Dataset camera calibration files
(containing intrinsic and extrinsic parameters) and publishing them on their
respective camera_info topics.
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor

from sensor_msgs.msg import CameraInfo

from pathlib import Path


# NOTE: This is a good reference:
# https://docs.ros.org/en/rolling/p/image_pipeline/camera_info.html for an intro
# as to what we're trying to publish here.

class CameraInfoNode(Node):

    def __init__(self):
        super().__init__('camera_info_node')
        self.declare_parameter(name="camera_parameters_filepath",
                               value="",
                               descriptor=ParameterDescriptor(description="Full path to the left camera calibration file as provided in the EuRoC MAV Dataset ASL files."))
        self.declare_parameter(name="camera_frame_id",
                               value="camera_link",
                               descriptor=ParameterDescriptor(description="The frame_id of the camera link that the camera_info corresponds with."))

        # Check the provided parameter filepath.
        # TODO: Maybe in the future, these types of checks should be the
        # responsibility of the launch file. I.e., perform those checks in the
        # launch file, and then just assume they're valid here?
        assert(self.get_parameter("camera_parameters_filepath").value != "")
        assert(Path(self.get_parameter("camera_parameters_filepath").value).exists())
        self.get_logger().info(f"Found valid configuration at provided path {self.get_parameter('camera_parameters_filepath').value}")

        # Try to open up the camera parameter yaml file. Read the configuration
        # into an instance variable if succesful.
        try:
            with open(self.get_parameter("camera_parameters_filepath").value, 'r') as file:
                self.camera_parameters = file.read()
                self.get_logger().info(f"Successfully read camera parameters from file.")
        except Exception as e:
            self.get_logger().error(f"Failed to read camera parameters from file: {e}")
            return
        
        # Publish the camera parameters on the camera_info topic.
        self.camera_info_publisher = self.create_publisher(CameraInfo, 'camera_info', 10)
        self.timer = self.create_timer(1.0, self.publish_camera_info)


    """
    # General sensor definitions.
    sensor_type: camera
    comment: VI-Sensor cam0 (MT9M034)

    # Sensor extrinsics wrt. the body-frame.
    T_BS:
    cols: 4
    rows: 4
    data: [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
            0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
            -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
            0.0, 0.0, 0.0, 1.0]

    # Camera specific definitions.
    rate_hz: 20
    resolution: [752, 480]
    camera_model: pinhole
    intrinsics: [458.654, 457.296, 367.215, 248.375] #fu, fv, cu, cv
    distortion_model: radial-tangential
    distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]

    """

    def publish_camera_info(self):
        # Create a CameraInfo message and populate it with the camera parameters.
        camera_info_msg = CameraInfo()
        camera_info_msg.header.frame_id = self.get_parameter("camera_frame_id").value
        camera_info_msg.height = self.camera_parameters["resolution"][1]
        camera_info_msg.width = self.camera_parameters["resolution"][0]
        camera_info_msg.distortion_model = "plumb_bob"
        # NOTE: per Kimera-VIO:
        # https://github.com/MIT-SPARK/Kimera-VIO?tab=readme-ov-file#3-parameters,
        # apparently you can set the 2nd distortion coefficient to 0 if you are
        # only given 4 coefficients?
        camera_info_msg.d = self.camera_parameters["distortion_coefficients"]
        # Insert the 0 for the 2nd distortion coefficient.
        camera_info_msg.d.insert(1, 0.0)
        camera_info_msg.k = [self.camera_parameters["intrinsics"][0], 0.0, self.camera_parameters["intrinsics"][2],
                             0.0, self.camera_parameters["intrinsics"][1], self.camera_parameters["intrinsics"][3],
                             0.0, 0.0, 1.0]
        
        # TODO: The "r" matrix will probably be identity or very close to it, as
        # this is the relative orietntation between the left camera and right
        # camera comprising the stereo pair. If being very correct, could
        # populate this with the rotation matrix we get from TF between the two
        # camera frames, but for now, just set it to identity.
        camera_info_msg.r = [1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0]

        # P is the projection matrix given by
        # P = K * [R | t]
        # where K is the camera matrix, R is the rotation matrix from the
        # "world" frame to the camera frame, and t is the translation vector in
        # the camera frame to the world frame origin. In the case of the left
        # camera in a stereo pair, R is the identity matrix and t is the zero.
        # For the right camera, R is still identity, but t is the negative of
        # the baseline between the two cameras. I.e., it is "where the left
        # camera is from the perspective of the right camera's frame."
        camera_info_msg.p = [458.654, 0.0, 367.215, 0.0,
                             0.0, 457.296, 248.375, 0.0,
                             0.0, 0.0, 1.0, 0.0]

        # Publish the camera info message.
        self.camera_info_publisher.publish(camera_info_msg)
        self.get_logger().info("Published camera info message.")


        # IDEA TO GET THIS GOING: Grab the camera frame to body trasnform and
        # publish that over tf. THen, once we also have the other camera's
        # sensor to body transform, we should be able to find (for the right
        # camera) the transform from left camera to the right camera == rotation
        # from left to right, and then the translation vector expressed from the
        # perspective of the right (==the destination frame). Just note the
        # documentation, as the baseline may have to be expressed in mm or
        # something like that.


        

def main(args=None):
    rclpy.init(args=args)
    camera_info_node = CameraInfoNode()
    rclpy.spin(camera_info_node)
    camera_info_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
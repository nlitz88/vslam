import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo

from message_filters import Subscriber, TimeSynchronizer

import cv2
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel


# import cv_bridge

class VoNode(Node):

    def __init__(self):
        super().__init__('vo_node')

        # self._left_image_sub = self.create_subscription(Image,
        #                                                 "/camera/infra1/image_rect_raw",
        #                                                 self.left_image_callback,
        #                                                 10)
        
        # self._depth_image_sub = self.create_subscription(Image,
        #                                                  "/camera/depth/image_rect_raw",
        #                                                  self.depth_image_callback,
        #                                                  10)

        # Create publisher for keypoint image.
        self._keypoint_image_pub = self.create_publisher(Image, "/keypoints", 10)

        # Create subscriber for CameraInfo messages.
        self._need_info = True
        self._left_camera_model = PinholeCameraModel()
        self._left_cam_info = self.create_subscription(msg_type=CameraInfo,
                                                  topic="/camera/infra1/camera_info",
                                                  callback=self.left_camera_info_callback,
                                                  qos_profile=10)

        # Create subscribers for left infrared camera and corresponding depth
        # image.
        self._left_image_sub = Subscriber(self, Image, "/camera/infra1/image_rect_raw")
        self._depth_image_sub = Subscriber(self, Image, "/camera/depth/image_rect_raw")

        # Using "message_filters" time sync feature to trigger a single callback
        # when both image messages with the same timestamp are received.
        queue_size = 10
        self.sync = TimeSynchronizer([self._left_image_sub, self._depth_image_sub], queue_size)
        self.sync.registerCallback(self.infra_depth_sync_callback)

        # Create cv_bridge instance.
        # Reference: https://automaticaddison.com/getting-started-with-opencv-in-ros-2-foxy-fitzroy-python/
        self.br = CvBridge()

        # Initiate ORB detector
        self.orb = cv2.ORB_create()
        # Create feature matcher.
        # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


    # def left_image_callback(self, left_image: Image):
    #     self.get_logger().debug(f"Received new left image with timestamp: {left_image.header.stamp}")

    # def depth_image_callback(self, depth_image: Image):
    #     self.get_logger().debug(f"Received new depth image with timestamp: {depth_image.header.stamp}")


    def infra_depth_sync_callback(self, left_image_msg: Image, depth_image_msg: Image):

        # If we haven't yet received camera parameters, bail out.
        if self._need_info:
            self.get_logger().warn("Received left and depth image, but haven't received camera parameters yet on camera info topic.")
            return

        self.get_logger().debug(f"Received new left infra image and depth image pair with matching timestamps: \
                                infra timestamp: {left_image_msg.header.stamp} \
                                depth timestamp: {depth_image_msg.header.stamp}")
        

        # OKAY--this is where we can start the visual odometry pipeline.
        # FOR NOW, going to start with the 3D-2D APPROACH, as according to
        # Davide's tutorial, will drift more slowly than the naive 3D-3D
        # approach I initially had in mind.

        # 1. Convert each image into opencv compatible images with cv_bridge.
        left_image = self.br.imgmsg_to_cv2(img_msg=left_image_msg)
        depth_image = self.br.imgmsg_to_cv2(img_msg=depth_image_msg)

        # 2. Extract ORB features from the left image. Starting from
        #    https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html
        #    This step will detect interesting keypoints that we will then
        #    generate ORB descriptors from.
        keypoints = self.orb.detect(left_image)
        # compute the descriptors with ORB
        keypoints, descriptors = self.orb.compute(left_image, keypoints)

        # print(f"Shape of keypoints: {keypoints}")
        print(f"Shape of descriptors: {descriptors.shape}")

        # DEBUG: Draw keypoints on image and publish.
        # draw only keypoints location,not size and orientation
        left_image_with_keypoints = cv2.drawKeypoints(left_image, keypoints, None, color=(0,255,0), flags=0)
        # Convert the debug image to a ROS image so we can publish it.
        left_image_keypoints_msg = self.br.cv2_to_imgmsg(cvim=left_image_with_keypoints,
                                                         encoding="rgb8")
        self._keypoint_image_pub.publish(left_image_keypoints_msg)

        # 3. Triangulate 3D position of each feature using stereo depth.
        #    I forgot; before we can do this, have to convert from 2D pixel
        #    coordinates to 2D image plane coordinates in millimeters. I think I
        #    need the camera's intrinsics to do this. I.e., the image plane
        #    offset and the pixel-size scale or something like that. I believe
        #    the focal length terms may also contain scale information.
        keypoint_positions = []
        print(f"First keypoint pos: {keypoints[0].pt}")
        for k, keypoint in enumerate(keypoints):

            
            # First, need to know what each of these keypoints looks like.
            # https://docs.opencv.org/4.x/d2/d29/classcv_1_1KeyPoint.html
            px = int(keypoint.pt[0])
            py = int(keypoint.pt[1])

            # Get depth from depth map at the keypoint position.
            Z = depth_image[py, px]
            print(f"Depth of keypoint {k} == {Z}")
            
            # Grab necessary camera intrinsic values from the "K" matrix.
            cx = self._left_camera_model.cx()
            cy = self._left_camera_model.cy()
            fx = self._left_camera_model.fx()
            fy = self._left_camera_model.fy()

            

            # X = (px - self.)

        #     x = (uv[0] - self.cx()) / self.fx()
        # y = (uv[1] - self.cy()) / self.fy()


            # Also need to divide by focal length.
            # AND THEN, once we have "image plane" coordinates, can multiply by
            # depth.
            # NEED TO KNOW if depth being provided in mm or meters--units of the
            # image plane coordinates will have to match.



        # If this is the first image,depth pair we're receiving, no
        # transformation to estimate, bail out.

        pass

    # https://answers.ros.org/question/393979/projecting-3d-points-into-pixel-using-image_geometrypinholecameramodel/
    def left_camera_info_callback(self, left_camera_info_msg):

        if self._need_info:
            self.get_logger().info("Ingesting new camera info message!")
            self._left_camera_model.fromCameraInfo(left_camera_info_msg)
            self._need_info = False


    # What I roughly need to do for monocular VO:

    # 1. Use cv_bridge to take ROS images and get them into a format we can work
    #    with in opencv.

    # 2. Use opencv to find features and generate ORB descriptors for them. I
    #    THINK this is a builtin function, we'll see.
    
    # 3. 

    
    # FOR STEREO VO:
    
    # 1. Maybe find features in monocular image? Then, using the extrinsics
    #    between RGB and infra cameras, find where those features land in the
    #    infrared images.
    # 2. THEN, can use stereo depth to identify...
    # ACTUALLY
    # They already give us the depth image. Which image is that computed over?
    # One of the infrared views?

    # Could see what Stephen Ferro did in his implementation.

    # Basically, one way or another, the most basic implementation should: Track
    # features and their 3D positions, and then use that to estimate the
    # camera's motion between frames. I.e., the camera's motion should be the
    # negative of the features motion in 3D. Probably have to do some basic
    # outlier rejection--probably not trivial for me though.

    # 


    # OKAY, per Davide's VO tutorial, I think I am interested in using 3D-->3D
    # for motion estimation. I.e., triangulating the 3D positions of the
    # features from each frame and computing the difference between those
    # positions == the camera's transformation (also using RANSAC).

    # Only thing I don't love about this: How can we instead (eventually) set
    # this up as an optimization problem?

    # One thing he points out: Therefore, 3D-3D motion estimation methods will
    # drift much more quickly than 3D-2D and 2D-2D methods
    
    # Sounds like the reason being because: Each time you compute the 3D
    # positions via triangulation (whether trivially with a stereo pair or the
    # optimal way), there is uncertainty immediately itnroduced to your
    # trajectory estimate. AND, you are doing this at each timestep, so you are
    # accumulating more and more uncertainty at every step without any
    # mitigation?

    # Therefore, if you're going to do 3D to 3D (or maybe any method in
    # general), it is important to select KEY FRAMES. I.e., only updating the
    # pose once sufficient parallax or change in the scene (==movement of the
    # camera) has occurred.

    # He goes on to say that in general





    # Although 3D-3D will drift pretty quickly (especially without keyframing),
    # I think it'll be a simple starting point--or at least a method that I can
    # demonstrate is not good and should not be pursued further.


    # 1. Create subscriber for left stereo image (as it corresponds with depth
    #    map).
    # 2. Create subscriber for depth image.
    #           DONE: but TODO: Figure out how to store them together.
    #           https://docs.ros.org/en/rolling/p/message_filters/Tutorials/Writing-A-Time-Synchronizer-Python.html
    # 3. TODO: Figure out how to synchronize those two (I.e., the left infra
    #    image with its corresponding depth map).
    # 4. Extract ORB features from the left image.
    # 5. Use this current image's disparity map to triangulate the 3D position
    #    of that point and store this.
    # 6. Use openCV guide to match these features to features detected in the
    #    previous left image.
    # 7. For each point that we have found a match for, basically want to add
    #    this to a collection of 3D point pairs. I.e., 
    # 8. Use estimateAffine3D to use RANSAC method to estimate the 3D
    #    transformation between the points at the current timestep and the
    #    points at the previous timestep:
    #    https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga396afb6411b30770e56ab69548724715
    #    NOTE: Not totally sure if this function is actually doing what I think.
    #    NOTE: Would iterative closest point be the more appropriate choice? How
    #    different is that from the approach
    #    https://vnav.mit.edu/material/15-RANSAC-notes.pdf
    

    # NOTE: FOR WORKING DIRECTLY ON THE ISAAC ROS CONTAINER, USING REALSENSE ROS
    # 4.55.1, https://github.com/IntelRealSense/realsense-ros/tree/4.51.1
    # Some of the parameters are named differently!
    # ros2 launch realsense2_camera rs_launch.py enable_sync:=true
    # depth_module.profile:=848x480x60 enable_infra1:=true enable_color:=false

    # ros2 launch realsense2_camera rs_launch.py enable_sync:=true
    # depth_module.profile:=848x480x30 enable_infra1:=true enable_color:=false
    # depth_module.emitter_enabled:=false

    # NOTE: For some reason, disabling the emitter at launch time isn't working,
    # so have to set the parameter later:
    # https://nvidia-isaac-ros.github.io/troubleshooting/hardware_setup.html#intel-realsense-camera-accidentally-enables-laser-emitter
    # ros2 param set /camera/camera depth_module.emitter_enabled 0
    

def main(args=None):
    rclpy.init(args=args)

    vo_node = VoNode()

    rclpy.spin(vo_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    vo_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
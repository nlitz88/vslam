import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo

from message_filters import Subscriber, TimeSynchronizer

import cv2
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel

import numpy as np


# From Google
def draw_keypoints_with_text(image, keypoints, positions):
    # Draw keypoints
    for i, keypoint in enumerate(keypoints):
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

        # Add text label
        text = str(f"[{positions[i][0]:.2f}, {positions[i][1]:.2f}, {positions[i][2]:.2f}]")
        cv2.putText(image, text, (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

    return image


class VoNode(Node):

    def __init__(self):
        super().__init__('vo_node')

        # Create publisher for keypoint image.
        self._keypoint_image_pub = self.create_publisher(Image, "/keypoints", 10)
        self._matched_points_image_pub = self.create_publisher(Image, "/matches", 10)

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

        # Store previous frame's images, features, and feature positions.
        self._last_left_frame = None
        self._last_keypoints = None
        self._last_descriptors = None
        self._last_positions = None
        self._last_tf = None


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
        # left_image = self.br.imgmsg_to_cv2(img_msg=left_image_msg)
        left_image = self.br.imgmsg_to_cv2(img_msg=left_image_msg, desired_encoding="rgb8")
        depth_image = self.br.imgmsg_to_cv2(img_msg=depth_image_msg)

        # 2. Extract ORB features from the left image. Starting from
        #    https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html
        #    This step will detect interesting keypoints that we will then
        #    generate ORB descriptors from.
        keypoints = self.orb.detect(left_image)
        # compute the descriptors with ORB
        keypoints, descriptors = self.orb.compute(left_image, keypoints)

        # DEBUG: Draw keypoints on image and publish.
        # draw only keypoints location,not size and orientation
        # left_image_with_keypoints = cv2.drawKeypoints(left_image, keypoints, None, color=(0,255,0), flags=0)
        # Convert the debug image to a ROS image so we can publish it.
        # left_image_keypoi                                                  nts_msg = self.br.cv2_to_imgmsg(cvim=left_image_with_keypoints,
        #encoding="rgb8")
        # self._keypoint_image_pub.publish(left_image_keypoints_msg)

        # 3. Triangulate 3D position of each feature using stereo depth.
        #    I forgot; before we can do this, have to convert from 2D pixel
        #    coordinates to 2D image plane coordinates in millimeters. I think I
        #    need the camera's intrinsics to do this. I.e., the image plane
        #    offset and the pixel-size scale or something like that. I believe
        #    the focal length terms may also contain scale information.
        keypoint_2d_positions = []  # DON'T KNOW IF WE NEED TO KEEP THIS
        keypoint_3d_positions = []  # DON'T KNOW IF WE NEED TO KEEP THIS
        for k, keypoint in enumerate(keypoints):

            
            # First, need to know what each of these keypoints looks like.
            # https://docs.opencv.org/4.x/d2/d29/classcv_1_1KeyPoint.html
            px = int(keypoint.pt[0])
            py = int(keypoint.pt[1])
            keypoint_2d_positions.append([px, py])

            # Get depth from depth map at the keypoint position.
            Z = depth_image[py, px]
            # keypoint_positions.append(Z)
            # print(f"Depth of keypoint {k} == {Z}")
            
            # Grab necessary camera intrinsic values from the "K" matrix.
            cx = self._left_camera_model.cx()
            cy = self._left_camera_model.cy()
            fx = self._left_camera_model.fx()
            fy = self._left_camera_model.fy()

            # Compute the 3D position of each of the keypoints.
            X = (px - cx)*Z / fx
            Y = (py - cy)*Z / fy

            keypoint_3d_positions.append([X, Y, Z])
        
        # Convert the position arrays into a numpy array.
        # keypoint_2d_positions = np.array(keypoint_2d_positions)
        keypoint_3d_positions = np.array(keypoint_3d_positions) # / 1000 # Convert to meters?? #
        # NOTE: Might have to "unconvert" these depending on what PnP.
        # NOTE: Do we have to convert to a numpy array?


        # points = zip(keypoints, keypoint_positions)

        # Publish keypoints with their respective 3D positions in the left
        # camera's frame. Note that this frame likely follows a different
        # convention from the ROS convention. Have to use a TF to get it into
        # the camera_link frame if we want that instead. TODO for later.
        left_image_with_keypoints = draw_keypoints_with_text(left_image, keypoints[:30], keypoint_3d_positions[:30])
        left_image_keypoints_msg = self.br.cv2_to_imgmsg(cvim=left_image_with_keypoints, encoding="rgb8")
        self._keypoint_image_pub.publish(left_image_keypoints_msg)
        



        # If this is the first set of images we're receiving, then there is no
        # transformation we can estimate--bail out. Before doing that, 
        if self._last_left_frame is None:
            self._last_left_frame = left_image
            self._last_keypoints = keypoints
            self._last_descriptors = descriptors
            self._last_positions = keypoint_3d_positions
            # At first timestep, transformation to first this camera's frame is just identity.
            self._last_tf = np.eye(4,4)
            return
        
        # OTHERWISE, do feature matching between the last image's features and
        # this images features.
        # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        # NOTE: The first parameter == the QUERY descriptors, which are compared
        # against the TRAIN descriptors == second argument.
        matches = self.matcher.match(descriptors, self._last_descriptors)
        # print(f"Number of matches: {len(matches)}")
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        # TEMP: Draw matches
        # Draw first 10 matches.
        matches_image = cv2.drawMatches(left_image,
                               keypoints,
                               self._last_left_frame,
                               self._last_keypoints,
                               matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        matches_image_msg = self.br.cv2_to_imgmsg(cvim=matches_image, encoding="rgb8")
        self._matched_points_image_pub.publish(matches_image_msg)

        # For the matches we find, our goal is to use 3D-2D correspondences and
        # use PNP to recover what the camera's R|t must have been based on where
        # the 3D points are in the previous timestep camera's frame and where we
        # are observing their projections in our current timestep camera image
        # plane.


        # What does PNP generally need? Basically, it should take N 2D pixel
        # coordinates (or image frame coords, not sure) from the CURRENT FRAME
        # and the 3D position of that feature FROM THE LAST FRAME. It will spit
        # out the R|t of the camera between the two frames.
        # https://en.wikipedia.org/wiki/Perspective-n-Point#EPnP
        # Maybe the opencv page will clarify this for me further:
        # https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
        # I think the cv::SOLVEPNP_EPNP is the underlying method we want, as it
        # works for a general collection of points--no coplanar requirement, for
        # example (like cv::SOLVEPNP_IPPE requires).
        # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e
        # documents SolvePnP RANSAC, which is what we'll use, and what Stephen
        # Ferro used in his approach.
        
        # Okay, the 3D points I need are going to be from the last frame. So
        # last_positions.
        # The image points are going to be the keypoints from this frame.
        # TODO: The only real tricky part here is: Using the indices returned
        # from the matching step to create pairs of 2D keypoints from this frame
        # to the 3D position of the matched feature from the previous frame.

        # https://docs.opencv.org/4.x/d4/de0/classcv_1_1DMatch.html#details
        # Each DMatch instance in the list of matches contains the index of the
        # query descriptor and the train descriptor. Also note that we sort the
        # matches by their distance (lowest best).

        # So, how can we use these?
        # For PnP, we want a list of 2D points and a parallel list of the 3D
        # positions of those identified points, but resolved in the previous
        # timestep camera frame.
        # What can I do to generate these lists?
        current_frame_feature_2d_points = []
        prev_frame_feature_3d_positions = []
        # Loop through these for now, come up with a faster way later.
        for m, match in enumerate(matches):
            current_frame_feature_idx = match.queryIdx
            prev_frame_feature_idx = match.trainIdx
            current_frame_feature_2d_points.append(keypoints[current_frame_feature_idx].pt)
            prev_frame_feature_3d_positions.append(self._last_positions[prev_frame_feature_idx]) # May have to make a numpy array out of this.

        current_frame_feature_2d_points = np.array(current_frame_feature_2d_points)
        prev_frame_feature_3d_positions = np.array(prev_frame_feature_3d_positions)

    
        # Now, with those lists of 2D features locations from the current frame
        # and the 3D positions of those features (expressed in the previous
        # camera frame), we can use PnP to solve for the R|t == SE(3)
        # transformation between these two frames. I.e., attitude and
        # translation.

        # Also need to grab Camera's instrict parameter matrix for PnP to use as
        # part of its measurement Jacobian matrix (I think).
        left_camera_intrinsics = self._left_camera_model.fullIntrinsicMatrix() # Gives us K == 3x3 matrix.
        left_camera_distortion = self._left_camera_model.distortionCoeffs()

        ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(prev_frame_feature_3d_positions,
                                                        current_frame_feature_2d_points,
                                                        left_camera_intrinsics,
                                                        left_camera_distortion)
        # rvecs == rotation as a "Rodrigues" vector? Use their functions to
        # convert this back to a rotation matrix.
        # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
        R = cv2.Rodrigues(rvecs)[0]
        t = tvecs
        # Form SE(3) "T" 4x4 matrix from R and t.
        T = np.zeros((4,4))
        T[0:3, 0:3] = R
        T[0:3, 3] = t.T
        T[3, 3] = 1

        # Now that we (in theory) have the transformation between these two
        # successive camera frames, we can obtain our running estimate of the
        # overall transformation from the first timestep's camera frame to the
        # current timestep's camera frame. Being that our poses == R|t are
        # SE(3), matrix multiplication is used to combine them. I.e., you
        # compose SE(3) elements by multiplying them together. This
        # transformation is what we intuitively think of as the robot's position
        # in the world frame whose origin is the camera frame at the first
        # timestep.
        self._last_tf = self._last_tf @ T
        self.get_logger().debug(f"New pose: {T}")


        # Once we're done processing the current frame, set it to the last.
        self._last_left_frame = left_image
        self._last_keypoints = keypoints
        self._last_descriptors = descriptors
        self._last_positions = keypoint_3d_positions



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

    # ros2 launch realsense2_camera rs_launch.py enable_sync:=true depth_module.profile:=848x480x30 enable_infra1:=true enable_color:=false depth_module.emitter_enabled:=false

    # NOTE: For some reason, disabling the emitter at launch time isn't working,
    # so have to set the parameter later:
    # https://nvidia-isaac-ros.github.io/troubleshooting/hardware_setup.html#intel-realsense-camera-accidentally-enables-laser-emitter
    # ros2 param set /camera/camera depth_module.emitter_enabled 0

    # ros2 bag record --storage mcap -a -x "(.*)theora(.*)|(.*)compressed(.*)"
    # for recording relevant topics. Have to 
    # sudo apt-get install ros-humble-rosbag2-storage-mcap
    

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
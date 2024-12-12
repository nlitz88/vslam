
import time
from typing import Optional
import cv2
import numpy as np
import rclpy
import matplotlib.pyplot as plt

from rclpy.node import Node
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
from message_filters import Subscriber, TimeSynchronizer
from tf2_ros import TransformBroadcaster
from transforms3d import quaternions, euler

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import UInt32, Int32

from vslam.ssc import ssc


# From Google
def draw_keypoints_with_text(image, keypoints, positions, color=(0, 255, 0)):
    # Draw keypoints
    for i, keypoint in enumerate(keypoints):
        
        x, y = int(keypoint[0]), int(keypoint[1])
        cv2.circle(image, (x, y), 4, color, -1)

        # Add text label
        text = str(f"{positions[i][2]:.2f}")
        cv2.putText(image, text, (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

    return image

# Function to generate a solid green image if the provided flag value is green,
# and a solid red image if the provided flag value is red. If a flag_name is
# provided, the text of the flag name will be displayed in the middle of the
# image in black.
def flag_to_image(flag: bool,
                  flag_name: Optional[str]) -> Image:
    """Generate a solid green or red image based on the provided flag value.
    
    Args:
        flag: A boolean flag value.
        flag_name: An optional string containing the name of the flag.

    Returns:
        An OpenCV image converted to a ROS Image message. The image will be red
        if the flag is False, and green if the flag is True. If a flag name is
        provided, the text of the flag name will be displayed in the middle of
        the image in black.
    """

    # Create a blank image
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Set the color based on the flag value
    color = (0, 255, 0) if flag else (0, 0, 255)
    image[:] = color

    # If a flag name is provided, add the text to the image
    if flag_name:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 0, 0)
        thickness = 2
        text_size = cv2.getTextSize(flag_name, font, font_scale, thickness)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = (image.shape[0] + text_size[1]) // 2
        cv2.putText(image, flag_name, (text_x, text_y), font, font_scale, font_color, thickness)

    # Convert the OpenCV image to a ROS Image message
    return CvBridge().cv2_to_imgmsg(image, encoding="bgr8")
    

# TODO: If we have time, turn this pipeline into a separate class that
# this ROS node class just calls into.
class VoNode(Node):

    def __init__(self):
        super().__init__('vo_node')

        # Declare any VO parameters.
        
        self.declare_parameter("orb_max_features", 1000)

        # Create publisher for keypoint image.
        self._keypoint_image_pub = self.create_publisher(Image, "keypoints", 10)
        self._keypoints_with_depth_image_pub = self.create_publisher(Image, "keypoints_with_depth", 10)
        self._matched_points_image_pub = self.create_publisher(Image, "matches", 10)
        self._odom_publisher = self.create_publisher(Odometry, "camera_odom", 10)
        self._filtered_depth_pub = self.create_publisher(Image, "filtered_depth", 10)
        self._inlier_count_pub = self.create_publisher(UInt32, "inlier_count", 10)
        self._correspondence_count_pub = self.create_publisher(UInt32, "correspondence_count", 10)
        self._bad_motion_est_pub = self.create_publisher(Image, "bad_motion_est", 10)
        self._inlier_outlier_image_pub = self.create_publisher(Image, "inlier_outlier_image", 10)
        self._inlier_depth_histogram_pub = self.create_publisher(Image, "inlier_depth_histogram", 10)
        self._outlier_depth_histogram_pub = self.create_publisher(Image, "outlier_depth_histogram", 10)

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Create subscriber for CameraInfo messages.
        self._received_intrinsics = False
        self._left_camera_model = PinholeCameraModel()
        self._left_cam_info = self.create_subscription(msg_type=CameraInfo,
                                                  topic="camera_info",
                                                  callback=self.left_camera_info_callback,
                                                  qos_profile=10)

        # Create subscribers for left infrared camera and corresponding depth
        # image.
        self._left_image_sub = Subscriber(self, Image, "left_image")
        self._depth_image_sub = Subscriber(self, Image, "depth_image")

        # Using "message_filters" time sync feature to trigger a single callback
        # when both image messages with the same timestamp are received.
        queue_size = 10
        self.sync = TimeSynchronizer([self._left_image_sub, self._depth_image_sub], queue_size)
        self.sync.registerCallback(self.infra_depth_sync_callback)

        # Create cv_bridge instance.
        # Reference: https://automaticaddison.com/getting-started-with-opencv-in-ros-2-foxy-fitzroy-python/
        self.br = CvBridge()

        # Initiate ORB detector
        self.orb = cv2.ORB_create(nfeatures=self.get_parameter("orb_max_features").value,
                                  edgeThreshold=10,
                                  fastThreshold=10,
                                  WTA_K=2,
                                  scoreType=cv2.ORB_HARRIS_SCORE)
        # Create feature matcher.
        # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Store previous frame's images, features, and feature positions.
        self._last_left_frame = None
        self._last_left_frame_with_keypoints = None
        self._last_keypoints = None
        self._last_descriptors = None
        self._last_positions = None
        # self._last_tf = None

        self.R = None
        self.t = None

        # Set up lookup table for gamma correction to "even out" image exposure.
        # https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
        gamma = 0.5
        self.lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            self.lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)


    def infra_depth_sync_callback(self, left_image_msg: Image, depth_image_msg: Image):

        # If we haven't yet received camera parameters, bail out.
        if not self._received_intrinsics:
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
        # print(f"depth image type: {depth_image.dtype}") # Remember, it's a
        # numpy array!

        # Perform gamma correction on the left image before detecting keypoints.        
        left_image = cv2.LUT(left_image, self.lookUpTable)

        # Perform bilateral filtering on the depth image.
        # depth_image = cv2.convertScaleAbs(depth_image)
        # depth_image = cv2.bilateralFilter(depth_image,9,25,25)
        # self._filtered_depth_pub.publish(self.br.cv2_to_imgmsg(depth_image)

        # 2. Extract ORB features from the left image. Starting from
        #    https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html
        #    This step will detect interesting keypoints that we will then
        #    generate ORB descriptors from.
        keypoints = self.orb.detect(left_image) 
        # compute the descriptors with ORB
        keypoints, descriptors = self.orb.compute(left_image, keypoints)

        # TODO: FILTER KEYPOINTS USING ANMS. See
        # https://github.com/BAILOOL/ANMS-Codes/blob/master/Python/ssc.py
        # FIRST: SORT KEYPOINTS BY RESPONSE.
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
        keypoints = ssc(keypoints, num_ret_points=100, tolerance=0.1, cols=left_image.shape[1], rows=left_image.shape[0])


        # DEBUG: Draw keypoints on image and publish.
        # draw only keypoints location,not size and orientation
        # left_image_with_keypoints = cv2.drawKeypoints(left_image, keypoints, None, color=(0,255,0), flags=0)
        # # Convert the debug image to a ROS image so we can publish it.
        # left_image_keypoints_msg = self.br.cv2_to_imgmsg(cvim=left_image_with_keypoints, encoding="rgb8")
        # self._keypoint_image_pub.publish(left_image_keypoints_msg)

        # 3. Triangulate 3D position of each feature using stereo depth.
        #    I forgot; before we can do this, have to convert from 2D pixel
        #    coordinates to 2D image plane coordinates in millimeters. I think I
        #    need the camera's intrinsics to do this. I.e., the image plane
        #    offset and the pixel-size scale or something like that. I believe
        #    the focal length terms may also contain scale information.
        # NOTE: I make these lists here so I can "filter out" the keypoints (and
        # their corresponding descriptors) that correspond with an invalid (0)
        # depth!
        keypoint_2d_positions = []  # DON'T KNOW IF WE NEED TO KEEP THIS
        keypoint_descriptors = []
        keypoint_3d_positions = []  # DON'T KNOW IF WE NEED TO KEEP THIS
        # print(f"Type of keypoints: {type(keypoints)}, each: {type(keypoints[0])}")
        # print(f"Type of descriptors: {type(descriptors)}, each: {type(descriptors[0])}")
        for k, keypoint in enumerate(keypoints):

            
            # First, need to know what each of these keypoints looks like.
            # https://docs.opencv.org/4.x/d2/d29/classcv_1_1KeyPoint.html
            px = int(keypoint.pt[0])
            py = int(keypoint.pt[1])

            # Get depth from depth map at the keypoint position.
            Z = depth_image[py, px]

            # if (Z < 100 or Z > 2000):
            #     continue
            # If the depth IS valid, then keep the keypoint by adding it to the
            # new list of keypoints.
            keypoint_2d_positions.append(keypoint)
            keypoint_descriptors.append(descriptors[k])

            # TODO: Figure out a way to characterize the variance of the range
            # measurement error at the given range, as we will eventually need
            # this in the sensor model we use for the realsense!

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
        keypoints = tuple(keypoint_2d_positions)
        descriptors = np.array(keypoint_descriptors)
        keypoint_3d_positions = np.array(keypoint_3d_positions)
        # NOTE: Might have to "unconvert" these depending on what PnP.
        # NOTE: Do we have to convert to a numpy array?
        # TODO: FILTER OUT KEYPOINTS WHOSE DEPTH == 0!

        # Print keypoints AFTER filtering.
        left_image_with_keypoints = cv2.drawKeypoints(left_image, keypoints, None, color=(0,255,0), flags=0)
        # Convert the debug image to a ROS image so we can publish it.
        left_image_keypoints_msg = self.br.cv2_to_imgmsg(cvim=left_image_with_keypoints, encoding="rgb8")
        self._keypoint_image_pub.publish(left_image_keypoints_msg)


        # Publish keypoints with their respective 3D positions in the left
        # camera's frame. Note that this frame likely follows a different
        # convention from the ROS convention. Have to use a TF to get it into
        # the camera_link frame if we want that instead. TODO for later.
        # left_image_with_keypoints_measured = draw_keypoints_with_text(left_image, keypoints[:20], keypoint_3d_positions[:20])
        # left_image_keypoints_measured_msg = self.br.cv2_to_imgmsg(cvim=left_image_with_keypoints_measured, encoding="rgb8")
        # self._keypoint_image_pub.publish(left_image_keypoints_measured_msg)
        

        # If this is the first set of images we're receiving, then there is no
        # transformation we can estimate--bail out. Before doing that, 
        if self._last_left_frame is None:
            self._last_left_frame = left_image
            self._last_left_frame_with_keypoints = left_image_with_keypoints
            self._last_keypoints = keypoints
            self._last_descriptors = descriptors
            self._last_positions = keypoint_3d_positions

            # TODO: Publish initial transformation here. 

            # ACTUALLY: Maybe this node should instead publish the transform
            # from C0-->CN, where CN is just the current timestep camera frame.

            # THEN, we would just define a static transform between the
            # odom frame and C0.



            # from odom to optical: 90 degree roll, -90 pitch?

            
            # While I don't want to get too caught up on this, what would this
            # really be publishing in practice?

            # In practice, an odometry source would be publishing "an odometry
            # estimate message" == what it thinks the pose + twist of the robot
            # is at each timestep. Then, you might combine the odometry
            # estimates of multiple odometry sources into a single optimization
            # problem to come up with a fused motion estimate (that reconciles
            # the error between them all).

            # In this way, we would not be publishing the transform between each
            # camera frame, or even the camera frame at each step. Instead, more
            # likely, I think we would be computing the transformation between
            # each base link frame--I.e., the frame this is rigidly attached to.
            # Then, we can publish the odometry of the base link rather than
            # just the camera frame.

            # The only wrinkle/caveat with this is dealing with the twist part
            # (==the linear and angular velocity estimate). The
            # position+orientation is trivial--as we just use the static
            # transform between the base link and the camera frame. This is
            # fine. What's NOT trivial is how to resolve/interpret the camera
            # frame's linear and angular velocity and compute from that the
            # velocity of the base link frame. This might literally just be the
            # kinematic transport theorem, but I need to read up on this.


            # In any case, a parameter of this node would be which frame we
            # would want to publish the odometry of. Then, this node would look
            # up a static transform between the base_link and the sensor frame
            # (camera_link in this case). We would compute the relative
            # transform between the two camera frames as usual. Then, compute
            # the camera_link pose w.r.t. the odom frame, and then use the
            # static transform between base link and camera link to compute
            # where the base link must be?

            # Maybe for now, 


            # We get the base_link pose in the odom frame == which is also just
            # the SE(3) transformation from the base_link frame to the odom
            # frame (because the translation vector is always in the destination
            # frame). Can take this and compute what the current camera_link to
            # odom frame transformation is by composing that base_link->odom
            # transformation with the camera_link-->base_link transformation.
            # This gives us camera_link to odom. We the compute the relative
            # transform from camera_link at time k-1 to camera_link at time k
            # (what pnp gives us). We can then find the transform from
            # camera_link to odom by composing the camera_link at time k to odom
            # with camera_link k-1 to k to get camera_link k to odom == new
            # camera pose in odom frame. Can then extract the new base link to
            # odom transform by composing camera_link_k to odom with base link
            # to camera_link_k (a static tf). This gives us the transform from
            # base_link to odom, which is == the base link's pose in the odom
            # frame. NEED TO VERIFY THE DETAILS. This logic should all be
            # isolated from the core VO functionalityi, though.





            # At first timestep, transformation to first this camera's frame is just identity.
            # self._last_tf = np.eye(4,4)
            # self.publish_camera_transform(child_frame=left_image_msg.header.frame_id,
            #                           parent_frame="odom",
            #                           transform=self._last_tf)
            # self.R = np.eye(3,3)
            self.t = np.zeros((3,1), dtype=float)

            # For now, just hardcode the initial orientation between the
            # camera_link at timestep 0 and the odom frame.
            self.R = euler.euler2mat(np.deg2rad(90), np.deg2rad(-90), 0)
            # Publish a transformation with these.
            self.publish_camera_transform(child_frame=left_image_msg.header.frame_id,
                                          parent_frame="odom",
                                          rotation=self.R,
                                          translation=self.t)

            return
        
        # OTHERWISE, do feature matching between the last image's features and
        # this images features.
        # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        # NOTE: The first parameter == the QUERY descriptors, which are compared
        # against the TRAIN descriptors == second argument.
        matches = self.matcher.match(descriptors, self._last_descriptors)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        # Draw first 10 matches.
        # matches_image = cv2.drawMatches(left_image_with_keypoints,
        #                                 keypoints,
        #                                 self._last_left_frame_with_keypoints,
        #                                 self._last_keypoints,
        #                                 matches[:20],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # matches_image_msg = self.br.cv2_to_imgmsg(cvim=matches_image, encoding="rgb8")
        # self._matched_points_image_pub.publish(matches_image_msg)

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

        # https://docs.opencv.org/4.x/d4/de0/classcv_1_1DMatch.html#details
        # Each DMatch instance in the list of matches contains the index of the
        # query descriptor and the train descriptor. Also note that we sort the
        # matches by their distance (lowest best).

        # So, how can we use these?
        # For PnP, we want a list of 2D points and a parallel list of the 3D
        # positions of those identified points, but resolved in the previous
        # timestep camera frame.
        current_frame_feature_2d_points = []
        prev_frame_feature_3d_positions = []
        # Loop through these for now, come up with a faster way later.
        for m, match in enumerate(matches):
            current_frame_feature_idx = match.queryIdx
            prev_frame_feature_idx = match.trainIdx
            current_frame_feature_2d_points.append(keypoints[current_frame_feature_idx].pt)
            prev_frame_feature_3d_positions.append(self._last_positions[prev_frame_feature_idx]) # May have to make a numpy array out of this.
        # TEMP: Remove
        # if len(prev_frame_feature_3d_positions) < 8:
        #     return
        # Convert the resulting lists to numpy arrays.
        current_frame_feature_2d_points = np.array(current_frame_feature_2d_points)
        prev_frame_feature_3d_positions = np.array(prev_frame_feature_3d_positions)

        self.get_logger().debug(f"Number of matches: {len(matches)}")

        # Publish keypoints with their depth values.
        left_image_with_depth_keypoints = draw_keypoints_with_text(left_image, current_frame_feature_2d_points, prev_frame_feature_3d_positions, color=(0, 255, 0))
        left_image_with_depth_keypoints_msg = self.br.cv2_to_imgmsg(cvim=left_image_with_depth_keypoints, encoding="bgr8")
        self._keypoints_with_depth_image_pub.publish(left_image_with_depth_keypoints_msg)

    
        # Now, with those lists of 2D features locations from the current frame
        # and the 3D positions of those features (expressed in the previous
        # camera frame), we can use PnP to solve for the R|t == SE(3)
        # transformation between these two frames. I.e., attitude and
        # translation.
        # Also need to grab Camera's instrict parameter matrix for PnP to use as
        # part of its measurement Jacobian matrix (I think).
        left_camera_intrinsics = self._left_camera_model.fullIntrinsicMatrix() # Gives us K == 3x3 matrix.
        left_camera_distortion = self._left_camera_model.distortionCoeffs()

        try:
            # TODO: Tune PnPRansac parameters!
            ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(prev_frame_feature_3d_positions,
                                                            current_frame_feature_2d_points,
                                                            left_camera_intrinsics,
                                                            left_camera_distortion,
                                                            useExtrinsicGuess=False,
                                                            iterationsCount=100,
                                                            reprojectionError=8,
                                                            confidence=0.90)
        except Exception as exc:
            self.get_logger().warn(f"solvePnPRansac failed with exception:\n{exc}")
            return
        
        if not ret:
            self.get_logger().warn("solvePnPRansac failed to converge.")
            return
        
        # TODO: REMOVE DEBUGGING LINES BELOW.
        tvecs = tvecs / 1000 # Convert to meters from mm -- scaling down even more so I can see the progression in RVIZ.
        euler_angles = euler.mat2euler(cv2.Rodrigues(rvecs)[0])
        euler_angles_deg = [np.rad2deg(angle_rad) for angle_rad in euler_angles]

        self.get_logger().debug(f"\n\
                                  PnP Translation (x, y, z) : \n{tvecs}\n \
                                  PnP Rotation (r, p, y)    : \n{euler_angles_deg}\n \
                                  Number of point 3D-2D correspondences: {len(current_frame_feature_2d_points)}\n \
                                  Number of inlier correspondences     : {len(inliers)}\n \
                                  PnP Return status: {ret}")
        
        inliers = inliers.flatten()
        outlier_mask = np.ones(current_frame_feature_2d_points.shape[0], dtype=bool)
        outlier_mask[inliers] = False

        # FOR NOW: We will identify an erroneous or "bad" motion estimate via
        # its magnitude. Basically, the rotation and translation between frames
        # in this dataset should be tiny. Therefore, if the magnitude is large,
        # then something's wrong.
        bad_motion_est = False
        BAD_TRANSLATION_THRESHOLD = 0.4
        if np.linalg.norm(tvecs) > BAD_TRANSLATION_THRESHOLD:
            self.get_logger().warn(f"Warning: Translation growing rapidly! T vector is {tvecs}")
            bad_motion_est = True
            
            
        # NOW, if this is the case, what do we want to analyze?
        # 1. For both the inliers and outliers, create a histogram of the
        #    depth values. I want to see if the inliers are all valid depth
        #    values.
        inlier_depths = prev_frame_feature_3d_positions[inliers][:,2]
        outlier_depths = prev_frame_feature_3d_positions[outlier_mask][:,2]
        # Create a histogram where each bin is 1mm wide.
        bins = np.arange(0, 5000, 10)
        inlier_hist, _ = np.histogram(inlier_depths, bins=bins)
        outlier_hist, _ = np.histogram(outlier_depths, bins=bins)

        start = time.perf_counter()
        # Plot the inlier histogram.
        fig, ax = plt.subplots()
        ax.hist(inlier_depths, bins=bins, histtype='stepfilled', label="Inliers")
        ax.set_xlabel("Depth (mm)")
        ax.set_ylabel("Frequency")
        ax.legend()

        # Create an image from the inlier histogram and publish it.
        fig.canvas.draw()
        plt_img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt_img_msg = self.br.cv2_to_imgmsg(plt_img_np, encoding="rgb8")
        self._inlier_depth_histogram_pub.publish(plt_img_msg)
        plt.close(fig)

        # Plot the outlier histogram.
        fig, ax = plt.subplots()
        ax.hist(outlier_depths, bins=bins, histtype='stepfilled', label="Outliers")
        ax.set_xlabel("Depth (mm)")
        ax.set_ylabel("Frequency")
        ax.legend()

        # Create an image from the outlier histogram and publish it.
        fig.canvas.draw()
        plt_img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt_img_msg = self.br.cv2_to_imgmsg(plt_img_np, encoding="rgb8")
        self._outlier_depth_histogram_pub.publish(plt_img_msg)
        plt.close(fig)

        end = time.perf_counter()
        self.get_logger().warn(f"Time to plot histograms: {end-start}")



        # Publish a flag image indicating whether the motion estimate is good or
        # bad.
        flag_image = flag_to_image(flag=not bad_motion_est, flag_name="Good Motion Estimate")
        self._bad_motion_est_pub.publish(flag_image)

        # TODO: Visualize the inliers and outliers.
        # 1. Draw the inlier keypoints in blue, and the outlier points in red.
        # 2. Draw the inlier 3D depth alongside the inlier keypoints.
        # 3. Publish this image to a topic for debugging.
        
        # Get a numpy array of all the rows of the inliers.
        # self.get_logger().warn(f"Current shape of current_frame_feature_2d_points: {current_frame_feature_2d_points.shape}")
        # self.get_logger().warn(f"Shape of inliers: {inliers.shape}")
        
        # self.get_logger().warn(f"Inlier list: {inliers}")

        # The inlier list is a list of all the row indices from the
        # current_frame_feature_2d_points array that are inliers. Select these
        # rows.
        inlier_keypoints = current_frame_feature_2d_points[inliers]
        inlier_positions = prev_frame_feature_3d_positions[inliers]
        inlier_image = draw_keypoints_with_text(left_image, inlier_keypoints, inlier_positions)
        # Draw the outlier keypoints in red.
        outlier_keypoints = current_frame_feature_2d_points[outlier_mask]
        outlier_positions = prev_frame_feature_3d_positions[outlier_mask]
        combined_image = draw_keypoints_with_text(inlier_image, outlier_keypoints, outlier_positions, color=(0, 0, 255))

        # Publish the image with inliers and outliers.
        combined_image_msg = self.br.cv2_to_imgmsg(cvim=combined_image, encoding="bgr8")
        self._inlier_outlier_image_pub.publish(combined_image_msg)


        # inlier_keypoints = current_frame_feature_2d_points[inliers]
        # inlier_positions = prev_frame_feature_3d_positions[inliers]
        # self.get_logger().warn(f"Shape of inlier keypoints: {inlier_keypoints.shape}")
        # self.get_logger().warn(f"Shape of inlier positions: {inlier_positions.shape}")
        # inlier_image = draw_keypoints_with_text(left_image, inlier_keypoints, inlier_positions)
        # # Draw the outlier keypoints in red.
        # outlier_keypoints = current_frame_feature_2d_points[np.logical_not(inliers)]
        # outlier_positions = prev_frame_feature_3d_positions[np.logical_not(inliers)]
        # outlier_image = draw_keypoints_with_text(inlier_image, outlier_keypoints, outlier_positions, color=(0, 0, 255))



        # TODO: Visualize outlier matches.
        # 1. Draw the outlier keypoints in red.

        # matches_image = cv2.drawMatches(left_image_with_keypoints,
        #                                 keypoints,
        #                                 self._last_left_frame_with_keypoints,
        #                                 self._last_keypoints,
        #                                 matches[:20],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # matches_image_msg = self.br.cv2_to_imgmsg(cvim=matches_image, encoding="rgb8")
        # self._matched_points_image_pub.publish(matches_image_msg)

    
        
        # TODO: Take a look at the depth values associated with the inliers and
        # outliers! Are there any inliers with completely invalid depth values?
        # The point of doing this is to figure out what filters to add on the 3D
        # points that we use. I.e., if the depth is zero, this is not helpful to
        # us and should not be considered at all. However, if RANSAC is already
        # not considering these points, then we don't really need to worry about
        # filtering them?

        # TODO: Also between the problematic frames and the good frames, compare
        # the 2D characteristics of those features as well!
        # 
        # Also take a look at the 2D characteristics of the inliers and
        # outliers. I.e., are all the inliers concrentraed 

        # The key is figuring out what is happening (what the inliers and
        # inliers look like) on those frames where the transformation is
        # MASSIVE!

        # TODO: Also look at the outlier correspondences (visually). What can we
        # observe?
        
        # TODO: Also, it might be nice to look at 
        
        # Publish debugging transforms.

        # test_msg = UInt32()
        self._inlier_count_pub.publish(UInt32(data=len(inliers)))
        self._correspondence_count_pub.publish(UInt32(data=len(current_frame_feature_2d_points)))

        # tvecs[2] = 0
        # print(f"Translation: {tvecs}")
        # print(f"")
        # print(f"ret: {ret}")
        # if np.linalg.norm(tvecs) > 1:
        #     self.get_logger().warn(f"Warning: Translation growing rapidly! T vector is {tvecs}")
        #     return

        
        if (ret):
            # rvecs == rotation as a "Rodrigues" vector? Use their functions to
            # convert this back to a rotation matrix.
            # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
            R = cv2.Rodrigues(rvecs)[0]
            # R == Transform FROM k-1 to k. Need to transpose this to get k to k-1!!!!
            # self.R == Transform from k-1 to odom.

            self.R = self.R @ R.T
            self.t = self.t + self.R @ tvecs

            self.get_logger().info(f"New position: {self.t}")

            # # Form SE(3) "T" 4x4 matrix from R and t.
            # T = np.zeros((4,4))
            # T[0:3, 0:3] = R
            # T[0:3, 3] = t.T
            # T[3, 3] = 1

            # # Now that we (in theory) have the transformation between these two
            # # successive camera frames, we can obtain our running estimate of the
            # # overall transformation from the first timestep's camera frame to the
            # # current timestep's camera frame. Being that our poses == R|t are
            # # SE(3), matrix multiplication is used to combine them. I.e., you
            # # compose SE(3) elements by multiplying them together. This
            # # transformation is what we intuitively think of as the robot's position
            # # in the world frame whose origin is the camera frame at the first
            # # timestep.
            # self._last_tf = self._last_tf @ T
            # # self.get_logger().debug(f"New pose: {T}")
            # self.get_logger().info(f"New Position: {self._last_tf[0:3, 3]}")

            
            self.publish_camera_transform(child_frame="camera_link",
                                        parent_frame="odom",
                                        rotation=self.R,
                                        translation=self.t)
            

            # Once we're done processing the current frame, set it to the last.
            self._last_left_frame = left_image
            self._last_left_frame_with_keypoints = left_image_with_keypoints
            self._last_keypoints = keypoints
            self._last_descriptors = descriptors
            self._last_positions = keypoint_3d_positions

    # https://answers.ros.org/question/393979/projecting-3d-points-into-pixel-using-image_geometrypinholecameramodel/
    def left_camera_info_callback(self, left_camera_info_msg):

        if not self._received_intrinsics:
            self.get_logger().info("Ingesting new camera info message!")
            self._left_camera_model.fromCameraInfo(left_camera_info_msg)
            self._received_intrinsics = True

    def publish_camera_transform(self, child_frame, parent_frame, rotation, translation):

        # Publish new transform from odom to whatever frame this is?
        # MAYBE CHECK CAMERA_INFO OR THE IMAGE FIELD TO SEE WHAT FRAME IT SAYS
        # IT'S IN.


        t = TransformStamped()

        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        # TODO: Wait a minute--I think there's a problem here with the code I
        # had here for the rotation. I think I grabbed this from one of the ROS
        # tutorials, but didn't update it for the rotations--I was only only
        # grabbing a single angle?? Not sure yet. Maybe that's not related to
        # the issue. But definitely go back through this and make sure what
        # we're doing is reasonable. Probably have to go from rotation matrix to
        # quaternion (although, these both represent SO(3) rotations no?).

        # Turtle only exists in 2D, thus we get x and y translation
        # coordinates from the message and set the z coordinate to 0
        # t.transform.translation.x = translation[0,0]
        # t.transform.translation.y = translation[1,0]
        # t.transform.translation.z = translation[2,0]

        # TODO: Convert SO(3) rotation matrix to a quaternion.
        # NOTE: self.R is going to be the rotation FROM the camera_link to the
        # odom frame (I think). I think we are publishing the opposite of this,
        # right? I.e., odom --> camera_link? Therefore, take the tranpose of
        # self.R to get the rotation from odom to base link.
        odom_to_cam_link_q = quaternions.mat2quat(self.R.T)
        # Likewise, if we're publishing the transform from odom to camera_link,
        # then the translation vector needs to be expressed in the destination
        # frame == the camera_link. Therefore, use self.R.T to resolve the odom
        # frame translation that we have (self.t) into the camera_link frame and
        # then negate it. This gives us the translation vector to the odom frame
        # from the perspective of the camera link frame. See
        # https://motion.cs.illinois.edu/RoboticSystems/CoordinateTransformations.html
        # for reference on how to intuitively invert SE(3) transformations.
        odom_link_from_cam_link_t = -(self.R.T @ self.t)
        
        # NOTE: The translation vector is a 3x1 numpy vector/array.
        t.transform.translation.x = odom_link_from_cam_link_t[0,0]
        t.transform.translation.y = odom_link_from_cam_link_t[1,0]
        t.transform.translation.z = odom_link_from_cam_link_t[2,0]

        # NOTE: For transforms3d, the w term is at the zero index:
        # https://matthew-brett.github.io/transforms3d/reference/transforms3d.quaternions.html
        t.transform.rotation.w = odom_to_cam_link_q[0]
        t.transform.rotation.x = odom_to_cam_link_q[1]
        t.transform.rotation.y = odom_to_cam_link_q[2]
        t.transform.rotation.z = odom_to_cam_link_q[3]

        

        # For the same reason, turtle can only rotate around one axis
        # and this why we set rotation in x and y to 0 and obtain
        # rotation in z axis from the message
        # q = quaternion_from_euler(0, 0, msg.theta)
        # t.transform.rotation.x = q[0]
        # t.transform.rotation.y = q[1]
        # t.transform.rotation.z = q[2]
        # t.transform.rotation.w = q[3]
        # t.transform.rotation = 

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

        # # Create odometry message.
        # odom_msg = Odometry()
        # odom_msg.header.stamp = t.header.stamp
        # odom_msg.header.frame_id = parent_frame
        # odom_msg.pose.pose.position.x = translation[0,0]
        # odom_msg.pose.pose.position.y = translation[1,0]
        # odom_msg.pose.pose.position.z = translation[2,0]

        # self._odom_publisher.publish(odom_msg)


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
    vo_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
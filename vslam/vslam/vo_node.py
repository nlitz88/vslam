import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image

from message_filters import Subscriber, TimeSynchronizer


# import cv_bridge

class VoNode(Node):

    def __init__(self):
        super().__init__('vo_node')
        # self.subscription = self.create_subscription(
        #     String,
        #     'topic',
        #     self.listener_callback,
        #     10)
        # self.subscription  # prevent unused variable warning

        # self._left_image_sub = self.create_subscription(Image,
        #                                                 "/camera/infra1/image_rect_raw",
        #                                                 self.left_image_callback,
        #                                                 10)
        
        # self._depth_image_sub = self.create_subscription(Image,
        #                                                  "/camera/depth/image_rect_raw",
        #                                                  self.depth_image_callback,
        #                                                  10)

        self._left_image_sub = Subscriber(self, Image, "/camera/infra1/image_rect_raw")
        self._depth_image_sub = Subscriber(self, Image, "/camera/depth/image_rect_raw")

        queue_size = 10
        self.sync = TimeSynchronizer([self._left_image_sub, self._depth_image_sub], queue_size)
        self.sync.registerCallback(self.infra_depth_sync_callback)


    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

    def left_image_callback(self, left_image: Image):
        self.get_logger().debug(f"Received new left image with timestamp: {left_image.header.stamp}")

    def depth_image_callback(self, depth_image: Image):
        self.get_logger().debug(f"Received new depth image with timestamp: {depth_image.header.stamp}")


    def infra_depth_sync_callback(self, left_image: Image, depth_image: Image):
        self.get_logger().debug(f"Received new left infra image and depth image pair with matching timestamps: \
                                infra timestamp: {left_image.header.stamp} \
                                depth timestamp: {depth_image.header.stamp}")
        

        # OKAY--this is where we can start the visual odometry pipeline.

        pass


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
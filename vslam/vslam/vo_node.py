import time
from typing import Optional
import cv2
import numpy as np
import rclpy
import matplotlib
matplotlib.use('Agg') # FIX 4: Prevent matplotlib from blocking/leaking memory in ROS callbacks
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

# From Google
def draw_keypoints_with_text(image, keypoints, positions, color=(0, 255, 0)):
    for i, keypoint in enumerate(keypoints):
        x, y = int(keypoint[0]), int(keypoint[1])
        cv2.circle(image, (x, y), 4, color, -1)
        text = str(f"{positions[i][2]:.2f}")
        cv2.putText(image, text, (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
    return image

def flag_to_image(flag: bool, flag_name: Optional[str]) -> Image:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    color = (0, 255, 0) if flag else (0, 0, 255)
    image[:] = color
    if flag_name:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 0, 0)
        thickness = 2
        text_size = cv2.getTextSize(flag_name, font, font_scale, thickness)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = (image.shape[0] + text_size[1]) // 2
        cv2.putText(image, flag_name, (text_x, text_y), font, font_scale, font_color, thickness)
    return CvBridge().cv2_to_imgmsg(image, encoding="bgr8")
    
class VoNode(Node):
    def __init__(self):
        super().__init__('vo_node')

        self.declare_parameter("max_features", 1000)
        self.declare_parameter("quality_level", 0.01)
        self.declare_parameter("min_distance", 10)
        self.declare_parameter("block_size", 3)
        self.declare_parameter("lk_win_size", 15)
        self.declare_parameter("lk_max_level", 2)
        self.declare_parameter("lk_epsilon", 0.03)
        self.declare_parameter("lk_max_iterations", 10)
        self.declare_parameter("clahe_clip_limit", 2.0)
        self.declare_parameter("clahe_tile_grid_size", 8)
        self.declare_parameter("blur_kernel_size", 5)

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

        self.tf_broadcaster = TransformBroadcaster(self)

        self._received_intrinsics = False
        self._left_camera_model = PinholeCameraModel()
        self._left_cam_info = self.create_subscription(msg_type=CameraInfo,
                                                  topic="camera_info",
                                                  callback=self.left_camera_info_callback,
                                                  qos_profile=10)

        self._left_image_sub = Subscriber(self, Image, "left_image")
        self._depth_image_sub = Subscriber(self, Image, "depth_image")

        queue_size = 10
        self.sync = TimeSynchronizer([self._left_image_sub, self._depth_image_sub], queue_size)
        self.sync.registerCallback(self.infra_depth_sync_callback)

        self.br = CvBridge()

        self.max_features = self.get_parameter("max_features").value
        self.quality_level = self.get_parameter("quality_level").value
        self.min_distance = self.get_parameter("min_distance").value
        self.block_size = self.get_parameter("block_size").value
        
        lk_win_size = self.get_parameter("lk_win_size").value
        lk_max_level = self.get_parameter("lk_max_level").value
        lk_epsilon = self.get_parameter("lk_epsilon").value
        lk_max_iterations = self.get_parameter("lk_max_iterations").value
        
        self.lk_params = dict(winSize=(lk_win_size, lk_win_size),
                              maxLevel=lk_max_level,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, lk_max_iterations, lk_epsilon))
        
        # CLAHE and blur parameters
        clahe_clip_limit = self.get_parameter("clahe_clip_limit").value
        clahe_tile_grid_size = self.get_parameter("clahe_tile_grid_size").value
        blur_kernel_size = self.get_parameter("blur_kernel_size").value
        
        # Create CLAHE object
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size))
        
        # Blur kernel size must be odd
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1

        self._last_left_frame = None
        self._last_left_frame_with_keypoints = None
        self._last_keypoints = None
        self._last_descriptors = None
        self._last_positions = None

        self.R = None
        self.t = None

        gamma = 0.5
        self.lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            self.lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)


    def infra_depth_sync_callback(self, left_image_msg: Image, depth_image_msg: Image):
        if not self._received_intrinsics:
            self.get_logger().warn("Waiting for camera parameters...")
            return

        left_image = self.br.imgmsg_to_cv2(img_msg=left_image_msg, desired_encoding="rgb8")
        depth_image = self.br.imgmsg_to_cv2(img_msg=depth_image_msg)

        left_image = cv2.LUT(left_image, self.lookUpTable)
        
        # Apply CLAHE histogram equalization
        gray_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
        clahe_image = self.clahe.apply(gray_image)
        
        # Apply Gaussian blur for noise reduction
        blur_image = cv2.GaussianBlur(clahe_image, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        corners = cv2.goodFeaturesToTrack(blur_image, 
                                        maxCorners=self.max_features,
                                        qualityLevel=self.quality_level,
                                        minDistance=self.min_distance,
                                        blockSize=self.block_size)
        
        if corners is None:
            self.get_logger().warn("No corners detected")
            return
            
        keypoints = [cv2.KeyPoint(x=float(c[0][0]), y=float(c[0][1]), size=1) for c in corners]
        
        keypoint_2d_positions = []
        keypoint_3d_positions = []
        
        cx = self._left_camera_model.cx()
        cy = self._left_camera_model.cy()
        fx = self._left_camera_model.fx()
        fy = self._left_camera_model.fy()

        for k, keypoint in enumerate(keypoints):
            px, py = int(keypoint.pt[0]), int(keypoint.pt[1])
            Z = depth_image[py, px]

            if Z > 0:  
                keypoint_2d_positions.append(keypoint)
                X = (px - cx) * Z / fx
                Y = (py - cy) * Z / fy
                keypoint_3d_positions.append([X, Y, Z])
        
        keypoints = tuple(keypoint_2d_positions)
        keypoint_3d_positions = np.array(keypoint_3d_positions)
        
        left_image_with_keypoints = cv2.drawKeypoints(left_image, keypoints, None, color=(0,255,0), flags=0)
        self._keypoint_image_pub.publish(self.br.cv2_to_imgmsg(left_image_with_keypoints, encoding="rgb8"))

        if self._last_left_frame is None:
            self._last_left_frame = left_image
            self._last_left_frame_with_keypoints = left_image_with_keypoints
            self._last_keypoints = keypoints
            self._last_positions = keypoint_3d_positions

            self.t = np.zeros((3,1), dtype=float)
            self.R = euler.euler2mat(np.deg2rad(90), np.deg2rad(-90), 0)
            
            self.publish_camera_transform(child_frame=left_image_msg.header.frame_id,
                                          parent_frame="odom",
                                          rotation=self.R,
                                          translation=self.t)
            return
        
        current_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
        last_gray = cv2.cvtColor(self._last_left_frame, cv2.COLOR_RGB2GRAY)
        
        current_points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        last_points = np.array([kp.pt for kp in self._last_keypoints], dtype=np.float32)
        
        tracked_points, status, error = cv2.calcOpticalFlowPyrLK(
            last_gray, current_gray, last_points, None, **self.lk_params)
        
        # FIX 3: Flatten status array to prevent indexing errors
        mask = (status.flatten() == 1)
        good_new = tracked_points[mask]
        good_old = last_points[mask]
        good_3d_positions = self._last_positions[mask]
        
        tracked_image = left_image.copy()
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(tracked_image, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            cv2.circle(tracked_image, (int(a), int(b)), 5, (0, 0, 255), -1)
        
        self._matched_points_image_pub.publish(self.br.cv2_to_imgmsg(tracked_image, encoding="rgb8"))

        # FIX 5: Ensure arrays are contiguous float32 for solvePnPRansac
        current_frame_feature_2d_points = np.ascontiguousarray(good_new, dtype=np.float32)
        prev_frame_feature_3d_positions = np.ascontiguousarray(good_3d_positions, dtype=np.float32)

        left_image_with_depth_keypoints = draw_keypoints_with_text(left_image, current_frame_feature_2d_points, prev_frame_feature_3d_positions, color=(0, 255, 0))
        self._keypoints_with_depth_image_pub.publish(self.br.cv2_to_imgmsg(left_image_with_depth_keypoints, encoding="bgr8"))

        left_camera_intrinsics = self._left_camera_model.fullIntrinsicMatrix()
        left_camera_distortion = self._left_camera_model.distortionCoeffs()
        
        if left_camera_distortion is not None and np.all(left_camera_distortion == 0):
            left_camera_distortion = None

        try:
            ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(prev_frame_feature_3d_positions,
                                                            current_frame_feature_2d_points,
                                                            left_camera_intrinsics,
                                                            left_camera_distortion,
                                                            useExtrinsicGuess=False,
                                                            iterationsCount=100,
                                                            reprojectionError=8,
                                                            confidence=0.70)
        except Exception as exc:
            self.get_logger().warn(f"solvePnPRansac failed with exception:\n{exc}")
            return
        
        # FIX 5b: Safely check for None inliers
        if not ret or inliers is None:
            self.get_logger().warn(f"solvePnPRansac failed to converge or returned 0 inliers.")
            self._last_left_frame = left_image
            self._last_left_frame_with_keypoints = left_image_with_keypoints
            self._last_keypoints = keypoints
            self._last_positions = keypoint_3d_positions
            return
        
        tvecs = tvecs / 1000.0
        inliers = inliers.flatten()
        outlier_mask = np.ones(current_frame_feature_2d_points.shape[0], dtype=bool)
        outlier_mask[inliers] = False

        bad_motion_est = False
        BAD_TRANSLATION_THRESHOLD = 0.4
        if np.linalg.norm(tvecs) > BAD_TRANSLATION_THRESHOLD:
            self.get_logger().warn(f"Warning: Translation growing rapidly! T vector is {tvecs}")
            bad_motion_est = True
            
        inlier_depths = prev_frame_feature_3d_positions[inliers][:,2]
        outlier_depths = prev_frame_feature_3d_positions[outlier_mask][:,2]
        bins = np.arange(0, 5000, 10)

        fig, ax = plt.subplots()
        ax.hist(inlier_depths, bins=bins, histtype='stepfilled', label="Inliers")
        ax.set_xlabel("Depth (mm)")
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.canvas.draw()
        plt_img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self._inlier_depth_histogram_pub.publish(self.br.cv2_to_imgmsg(plt_img_np, encoding="rgb8"))
        plt.close(fig) # Closes silently since we are using 'Agg'

        fig, ax = plt.subplots()
        ax.hist(outlier_depths, bins=bins, histtype='stepfilled', label="Outliers")
        ax.set_xlabel("Depth (mm)")
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.canvas.draw()
        plt_img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self._outlier_depth_histogram_pub.publish(self.br.cv2_to_imgmsg(plt_img_np, encoding="rgb8"))
        plt.close(fig)

        self._bad_motion_est_pub.publish(flag_to_image(flag=not bad_motion_est, flag_name="Good Motion Estimate"))

        inlier_keypoints = current_frame_feature_2d_points[inliers]
        inlier_positions = prev_frame_feature_3d_positions[inliers]
        inlier_image = draw_keypoints_with_text(left_image, inlier_keypoints, inlier_positions)
        
        outlier_keypoints = current_frame_feature_2d_points[outlier_mask]
        outlier_positions = prev_frame_feature_3d_positions[outlier_mask]
        combined_image = draw_keypoints_with_text(inlier_image, outlier_keypoints, outlier_positions, color=(0, 0, 255))
        self._inlier_outlier_image_pub.publish(self.br.cv2_to_imgmsg(combined_image, encoding="bgr8"))

        self._inlier_count_pub.publish(UInt32(data=len(inliers)))
        self._correspondence_count_pub.publish(UInt32(data=len(current_frame_feature_2d_points)))

        if ret:
            R = cv2.Rodrigues(rvecs)[0]
            
            # FIX 1: Math sign error. Camera translation subtracts the global projection of the PnP frame movement.
            self.R = self.R @ R.T
            self.t = self.t - self.R @ tvecs 

            self.get_logger().info(f"New position: {self.t.flatten()}")

            self.publish_camera_transform(child_frame="camera_link",
                                        parent_frame="odom",
                                        rotation=self.R,
                                        translation=self.t)
            
            self._last_left_frame = left_image
            self._last_left_frame_with_keypoints = left_image_with_keypoints
            self._last_keypoints = keypoints
            self._last_positions = keypoint_3d_positions

    def left_camera_info_callback(self, left_camera_info_msg):
        if not self._received_intrinsics:
            self.get_logger().info("Ingesting new camera info message!")
            self._left_camera_model.fromCameraInfo(left_camera_info_msg)
            self._received_intrinsics = True

    # FIX 2: Completed transform publisher and corrected inverse logic.
    def publish_camera_transform(self, child_frame, parent_frame, rotation, translation):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        t_flat = translation.flatten()
        t.transform.translation.x = float(t_flat[0])
        t.transform.translation.y = float(t_flat[1])
        t.transform.translation.z = float(t_flat[2])

        # transforms3d returns [w, x, y, z]. ROS expects [x, y, z, w]
        q = quaternions.mat2quat(rotation)
        t.transform.rotation.w = float(q[0])
        t.transform.rotation.x = float(q[1])
        t.transform.rotation.y = float(q[2])
        t.transform.rotation.z = float(q[3])

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    # Initialize the ROS 2 Python client library
    rclpy.init(args=args)
    
    # Create an instance of your node
    vo_node = VoNode()
    
    try:
        # Keep the node alive and listening for callbacks
        rclpy.spin(vo_node)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        pass
    finally:
        # Clean up and shut down
        vo_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
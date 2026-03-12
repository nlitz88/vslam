import time
from typing import Optional
import cv2
import numpy as np
import rclpy
import matplotlib
matplotlib.use('Agg') 
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
from std_msgs.msg import UInt32

def draw_keypoints_with_text(image, keypoints, positions, color=(0, 255, 0)):
    for i, keypoint in enumerate(keypoints):
        # Handle both KeyPoint objects and numpy arrays
        pt = keypoint.pt if hasattr(keypoint, 'pt') else keypoint
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(image, (x, y), 4, color, -1)
        if i < len(positions):
            text = str(f"{positions[i][2]:.2f}")
            cv2.putText(image, text, (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return image

class VoNode(Node):
    def __init__(self):
        super().__init__('vo_node')

        # Parameters
        self.declare_parameter("max_features", 1000)
        self.declare_parameter("quality_level", 0.01)
        self.declare_parameter("min_distance", 10)
        self.declare_parameter("block_size", 3)
        self.declare_parameter("lk_win_size", 15)
        self.declare_parameter("lk_max_level", 2)
        self.declare_parameter("lk_epsilon", 0.03)
        self.declare_parameter("lk_max_iterations", 10)
        self.declare_parameter("max_depth_threshold", 10.0)

        # Publishers
        self._matched_points_image_pub = self.create_publisher(Image, "matches", 10)
        self._inlier_outlier_image_pub = self.create_publisher(Image, "inlier_outlier_image", 10)
        self._odom_publisher = self.create_publisher(Odometry, "camera_odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # State
        self._received_intrinsics = False
        self._left_camera_model = PinholeCameraModel()
        self.br = CvBridge()
        
        self._last_gray = None
        self._last_keypoints_2d = None # np.array of shape (N, 2)
        self._last_keypoints_3d = None # np.array of shape (N, 3)
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))

        # LK Params
        lks = self.get_parameter("lk_win_size").value
        self.lk_params = dict(winSize=(lks, lks),
                              maxLevel=self.get_parameter("lk_max_level").value,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                                        self.get_parameter("lk_max_iterations").value, 
                                        self.get_parameter("lk_epsilon").value))

        # Subscriptions
        self._left_cam_info = self.create_subscription(CameraInfo, "camera_info", self.left_camera_info_callback, 10)
        self._left_image_sub = Subscriber(self, Image, "left_image")
        self._depth_image_sub = Subscriber(self, Image, "depth_image")
        self.sync = TimeSynchronizer([self._left_image_sub, self._depth_image_sub], 10)
        self.sync.registerCallback(self.infra_depth_sync_callback)

    def extract_features(self, gray_img, depth_img):
        """Helper to find new features and their 3D positions."""
        corners = cv2.goodFeaturesToTrack(gray_img, 
                                         maxCorners=self.get_parameter("max_features").value,
                                         qualityLevel=self.get_parameter("quality_level").value,
                                         minDistance=self.get_parameter("min_distance").value)
        if corners is None: return None, None
        
        pts_2d, pts_3d = [], []
        cx, cy = self._left_camera_model.cx(), self._left_camera_model.cy()
        fx, fy = self._left_camera_model.fx(), self._left_camera_model.fy()

        for pt in corners:
            px, py = int(pt[0][0]), int(pt[0][1])
            if py >= depth_img.shape[0] or px >= depth_img.shape[1]: continue
            
            z = float(depth_img[py, px])
            if 0 < (z / 1000.0) <= self.get_parameter("max_depth_threshold").value:
                pts_2d.append([px, py])
                pts_3d.append([(px - cx) * z / fx, (py - cy) * z / fy, z])
        
        return np.array(pts_2d, dtype=np.float32), np.array(pts_3d, dtype=np.float32)

    def infra_depth_sync_callback(self, left_msg, depth_msg):
        if not self._received_intrinsics: return

        current_img = self.br.imgmsg_to_cv2(left_msg, "rgb8")
        current_gray = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY)
        depth_img = self.br.imgmsg_to_cv2(depth_msg, "passthrough")

        # --- INITIALIZATION ---
        if self._last_gray is None:
            self._last_keypoints_2d, self._last_keypoints_3d = self.extract_features(current_gray, depth_img)
            self._last_gray = current_gray
            return

        # --- FEATURE TRACKING (The Fix) ---
        # We track the OLD points into the CURRENT frame
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self._last_gray, current_gray, self._last_keypoints_2d, None, **self.lk_params)
        
        valid = status.flatten() == 1
        tracked_2d = next_pts[valid]
        tracked_3d_prev = self._last_keypoints_3d[valid]
        old_2d = self._last_keypoints_2d[valid]

        if len(tracked_2d) < 10:
            self.get_logger().warn("Lost tracks, re-initializing...")
            self._last_keypoints_2d, self._last_keypoints_3d = self.extract_features(current_gray, depth_img)
            self._last_gray = current_gray
            return

        # Visualize Tracks (Velocity Vectors)
        track_vis = current_img.copy()
        for new, old in zip(tracked_2d, old_2d):
            cv2.line(track_vis, tuple(old.astype(int)), tuple(new.astype(int)), (0, 255, 0), 2)
        self._matched_points_image_pub.publish(self.br.cv2_to_imgmsg(track_vis, "rgb8"))

        # --- MOTION ESTIMATION ---
        K = self._left_camera_model.fullIntrinsicMatrix()
        dist = self._left_camera_model.distortionCoeffs()
        
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(tracked_3d_prev, tracked_2d, K, dist, 
                                                     reprojectionError=4.0, confidence=0.99)

        if ret and inliers is not None:
            # Convert translation to meters (PnP returns units of the 3D points, which were mm)
            t_delta = tvec / 1000.0
            R_delta, _ = cv2.Rodrigues(rvec)

            # Update Global Pose: World_T_Cam = World_T_PrevCam * (Delta_T_Cam)^-1
            # Inverse of [R|t] is [R^T | -R^T * t]
            self.t = self.t - self.R @ (R_delta.T @ t_delta)
            self.R = self.R @ R_delta.T

            self.publish_camera_transform(left_msg.header.frame_id, "odom", self.R, self.t)

            # --- PREPARE FOR NEXT FRAME ---
            # Re-detect fresh features to keep the track count high
            self._last_keypoints_2d, self._last_keypoints_3d = self.extract_features(current_gray, depth_img)
            self._last_gray = current_gray

    def left_camera_info_callback(self, msg):
        if not self._received_intrinsics:
            self._left_camera_model.fromCameraInfo(msg)
            self._received_intrinsics = True

    def publish_camera_transform(self, child, parent, R, t):
        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = parent
        ts.child_frame_id = child
        
        ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = t.flatten()
        
        # FIX: transforms3d is [w, x, y, z] -> ROS is [x, y, z, w]
        q = quaternions.mat2quat(R)
        ts.transform.rotation.x = float(q[1])
        ts.transform.rotation.y = float(q[2])
        ts.transform.rotation.z = float(q[3])
        ts.transform.rotation.w = float(q[0])
        
        self.tf_broadcaster.sendTransform(ts)

def main(args=None):
    rclpy.init(args=args)
    node = VoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
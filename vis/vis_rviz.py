import cv2
import numpy as np
import yaml

import rospy
from std_msgs.msg import String
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, CompressedImage
import queue
from cv_bridge import CvBridge
import matplotlib.cm as cm

class Sensor(object):

    def __init__(self):
        rospy.init_node('lidar_camera_projection')
        print("Starting the node...")

        IMAGE_TOPIC = '/ecocar/stereo/left/image_raw/compressed'
        CLOUD_TOPIC = '/ecocar/ouster/lidar_packets'
        PROJECTED_TOPIC = '/lidar_points_projected'
        
        self.img_sub = rospy.Subscriber(IMAGE_TOPIC, CompressedImage, self.image_callback)
        self.pc_sub  = rospy.Subscriber(CLOUD_TOPIC, PointCloud2, self.lidar_callback)
        self.pro_pub = rospy.Publisher(PROJECTED_TOPIC, CompressedImage, queue_size=10)
        
        self.img_queue = queue.Queue()
        self.pc_queue = queue.Queue()


    def image_callback(self, msg):
        self.img_queue.put(msg)
    
    def lidar_callback(self, msg):
        # Process the lidar point cloud here
        self.pc_queue.put(msg)

    def img_get(self):
        img_msg = self.img_queue.get()
        img = CvBridge().compressed_imgmsg_to_cv2(img_msg)        
        
        return img

    def pc_get(self):
        pc_msg = self.pc_queue.get()
        pc = np.array(list(pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)), dtype=np.float32)
        import pdb; pdb.set_trace()
        return pc

    def is_img_queue_empty(self):
        return self.img_queue.empty()

    def is_pc_queue_empty(self):
        return self.pc_queue.empty()

def filter_pc_uvd(pc_uvd):
    width = 960
    height = 600
    pc_uvd_filtered = pc_uvd[(pc_uvd[:, 0] >= 0) & (pc_uvd[:, 0] < width) & (pc_uvd[:, 1] >= 0) & (pc_uvd[:, 1] < height) & (pc_uvd[:, 2] > 0)]
    
    return pc_uvd_filtered

def overlay_pc_on_img(img, pc_uvd_filtered):
    img_overlay = img.copy()
    for point in pc_uvd_filtered:
        x, y, _ = point
        cv2.circle(img_overlay, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    return img_overlay


def pixels_to_depth(pc_np, calib, IMG_H, IMG_W, return_depth=True, IMG_DEBUG_FLAG=False):
    """
    pc_np:      [N x >=3] point cloud in LiDAR frame
    image_pts   [N x uv]
    calib:      [dict] calibration dictionary
    IMG_W:      [int] image width
    IMG_H:      [int] image height

    Returns depth values in meters
    """
    lidar2camrect = calib['T_lidar_to_cam']

    # Remove points behind camera after coordinate system change
    pc_np = pc_np[:, :3].astype(np.float64) # Remove intensity and scale for opencv
    pc_homo = np.hstack((pc_np, np.ones((pc_np.shape[0], 1))))
    pc_rect_cam = (lidar2camrect @ pc_homo.T).T
    
    lidar_pts = pc_rect_cam / pc_rect_cam[:, -1].reshape(-1, 1)
    MAX_INT32 = np.iinfo(np.int32).max
    MIN_INT32 = np.iinfo(np.int32).min
    lidar_pts = np.clip(lidar_pts, MIN_INT32, MAX_INT32)
    lidar_pts = lidar_pts.astype(np.int32)[:, :2]

    import pdb; pdb.set_trace()

    pts_mask = pc_rect_cam[:, 2] > 1

    in_bounds = np.logical_and(
        np.logical_and(lidar_pts[:, 0]>=0, lidar_pts[:, 0]<IMG_W), 
        np.logical_and(lidar_pts[:, 1]>=0, lidar_pts[:, 1]<IMG_H)
    )

    valid_point_mask = in_bounds & pts_mask
    valid_lidar_points  = lidar_pts[valid_point_mask, :]
    valid_lidar_depth   = pc_rect_cam[valid_point_mask, 2] # Use z in cam frame
    
    # Sort lidar depths by farthest to closest
    sortidx = np.argsort(-valid_lidar_depth)
    valid_lidar_depth = valid_lidar_depth[sortidx]
    valid_lidar_points = valid_lidar_points[sortidx, :]

    if IMG_DEBUG_FLAG:
        test_img = np.zeros((IMG_H, IMG_W), dtype=int)
        test_img[valid_lidar_points[:, 1], valid_lidar_points[:, 0]] = 255
        cv2.imwrite("test.png", test_img)

    #1 Create LiDAR depth image
    depth_image_np = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    depth_image_np[valid_lidar_points[:, 1], valid_lidar_points[:, 0]] = valid_lidar_depth
    
    #2 Use cv2 di

    if IMG_DEBUG_FLAG:
        depth_mm = (depth_image_np * 1000).astype(np.uint16)
        cv2.imwrite("pp_depth_max.png", depth_mm)

    if return_depth:
        return depth_image_np


    return valid_lidar_points, valid_lidar_depth


def main():
    
    # Load the calibration files
    cam0_intrinsics_file = "./calibrations/44/calib_cam0_intrinsics.yaml"
    lidar2cam0_file = "./calibrations/44/calib_os1_to_cam0.yaml"

    with open(cam0_intrinsics_file, 'r') as f:
        cam_intrinsics = yaml.safe_load(f)

    with open(lidar2cam0_file, 'r') as f:
        lidar2cam0 = yaml.safe_load(f)

    K = np.array(cam_intrinsics["camera_matrix"]["data"]).reshape(3, 3)
    d = np.array(cam_intrinsics["distortion_coefficients"]["data"])
    T_lidar_to_cam = np.array(lidar2cam0["extrinsic_matrix"]["data"]).reshape(4, 4)
    # DUMMY = np.eye(3, 4, dtype=np.float32)

    # T_lidar2pixel = K @ DUMMY @ T_lidar2cam0 # (3,3) @ (3,4) @ (4,4) = (3,4)


    K = np.array(cam_intrinsics['camera_matrix']['data']).reshape(cam_intrinsics['camera_matrix']['rows'], cam_intrinsics['camera_matrix']['cols'])
    T_canon = np.zeros((3, 4))
    T_canon[:3, :3] = np.eye(3)
    T_lidar_to_pixels = K @ T_canon @ T_lidar_to_cam

    calib_dict = {'T_lidar_to_cam': T_lidar_to_pixels}
    
    sensor = Sensor()

    i = 0
    while not rospy.is_shutdown():
        
        img = None
        pc  = None

        while sensor.is_img_queue_empty() or sensor.is_pc_queue_empty():
            continue
        
        print("proceeding ...")

        img = sensor.img_get()
        pc = sensor.pc_get()

        # Project lidar points to image plane
        # np_ones = np.ones((pc.shape[0], 1), dtype=np.float32)
        # pc = np.hstack((pc, np_ones))
        # pc_uvd = np.matmul(T_lidar2pixel, pc.T).T  # (4,4) @ (4, N) = (4, N).T = (N, 4)
        # pc_uvd_filtered = filter_pc_uvd(pc_uvd)

        valid_points, valid_depths = pixels_to_depth(pc, calib_dict, 960, 600, return_depth=False, IMG_DEBUG_FLAG=True)

        # Color image with depth
        valid_z_map = np.clip(valid_depths, 1, 80)
        norm_valid_z_map = valid_z_map / max(valid_z_map)
        color_map = cm.get_cmap("Blues")(norm_valid_z_map) * 255 # [0,1] to [0, 255]]

        for pt_idx, pt in enumerate(valid_points):
            if color_map[pt_idx].tolist()==[0,0,0]:
                continue # Only circle non background
            img_np = cv2.circle(img, (pt[0], pt[1]), radius=2, color=color_map[pt_idx].tolist(), thickness=-1)

        cv2.imwrite(f"vis/color_depth_{i}.png", img_np)
# 
        # img_overlay = overlay_pc_on_img(img, pc_uvd_filtered)

        # cv2.imwrite(f'vis/overlay_image_{i}.png', img_overlay)
        i += 1
        # Convert the overlay image to CompressedImage message
        # img_msg = CvBridge().cv2_to_compressed_imgmsg(img_overlay, dst_format='png')
        # img_msg.header.stamp = rospy.Time.now()
        # img_msg.header.frame_id = "projected"

        # sensor.pro_pub.publish(img_msg)
        # rospy.loginfo("Image published")
        rospy.sleep(0.01)

if __name__ == "__main__":
    main()

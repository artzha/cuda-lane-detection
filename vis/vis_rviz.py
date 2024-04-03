import cv2
import numpy as np
import yaml

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, CompressedImage
import queue

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
        np_arr = np.frombuffer(img_msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        return img

    def pc_get(self):
        pc_msg = self.pc_queue.get()
        pc = np.frombuffer(pc_msg.data, dtype=np.float64).reshape(-1, 4)
        
        return pc[:, :3] # only (x, y, z)

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


def main():
    
    # Load the calibration files
    cam0_intrinsics_file = "/home/jihwan98/cuda-lane-detection/calibrations/44/calib_cam0_intrinsics.yaml"
    lidar2cam0_file = "/home/jihwan98/cuda-lane-detection/calibrations/44/calib_os1_to_cam0.yaml"

    with open(cam0_intrinsics_file, 'r') as f:
        cam0_intrinsics = yaml.safe_load(f)

    with open(lidar2cam0_file, 'r') as f:
        lidar2cam0 = yaml.safe_load(f)

    K_cam0 = np.array(cam0_intrinsics["camera_matrix"]["data"]).reshape(3, 3)
    T_lidar2cam0 = np.array(lidar2cam0["extrinsic_matrix"]["data"]).reshape(4, 4)
    DUMMY = np.eye(3, 4)
    T_lidar2pixel = K_cam0 @ DUMMY @ T_lidar2cam0 # (3,3) @ (3,4) @ (4,4) = (3,4)
    
    sensor = Sensor()

    while not rospy.is_shutdown():
        
        img = None
        pc  = None

        while sensor.is_img_queue_empty() or sensor.is_pc_queue_empty():
            continue
        
        print("proceeding ...")

        img = sensor.img_get()
        pc = sensor.pc_get()

        # Project lidar points to image plane
        np_ones = np.ones((pc.shape[0], 1))
        pc = np.hstack((pc, np_ones))
        pc_uvd = (T_lidar2pixel @ pc.T).T # (4,4) @ (4, N) = (4, N).T = (N, 4)
        pc_uvd_filtered = filter_pc_uvd(pc_uvd)

        img_overlay = overlay_pc_on_img(img, pc_uvd_filtered)
        
        # Publish the image with projected points
        img_msg = CompressedImage()
        img_msg.header.stamp = rospy.Time.now()
        img_msg.format = "png"
        img_msg.data = np.array(cv2.imencode('.png', img_overlay)[1]).tobytes()
        
        sensor.pro_pub.publish(img_msg)

        rospy.spin()

if __name__ == "__main__":
    main()
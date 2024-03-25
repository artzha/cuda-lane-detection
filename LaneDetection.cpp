#include "commons.h"
#include "HoughTransform.h"
#include "Preprocessing.h"
#include <time.h>
#include <iomanip>
#include <queue>

#include <yaml-cpp/yaml.h>

#include "ros/ros.h"
#include "sensor_msgs/CompressedImage.h"
#include "sensor_msgs/PointCloud2.h"

extern void convert_lines_to_uvd(
    const vector<Line>& lines, 
    const sensor_msgs::PointCloud2::ConstPtr cloud_msg, 
    const cv::Mat& T_lidar2pixel, 
    const size_t height,
    const size_t width,
    vector<LineAnchors> &uvd);
extern void convert_uvd_to_xyz(
    const vector<LineAnchors>& uvd,
    const cv::Mat& T_uvd2xyz,
    vector<LineAnchors>& xyz
);
extern void detectLanes(VideoCapture inputVideo, VideoWriter outputVideo, int houghStrategy);
extern vector<Line> detectLanes(sensor_msgs::CompressedImage::ConstPtr msg, HoughTransformHandle* handle, VideoWriter &outputVideo, int houghStrategy);
extern void drawLines(Mat &frame, vector<Line> lines);
extern Mat plotAccumulator(int nRows, int nCols, int *accumulator);

// Function to convert a vector to a cv::Mat
extern cv::Mat vectorToMat(const std::vector<double>& vec, int rows, int cols) {
    cv::Mat mat(rows, cols, CV_64F); // Assuming double precision; adjust the type if necessary
    std::memcpy(mat.data, vec.data(), vec.size() * sizeof(double));
    return mat;
}

std::string IMAGE_TOPIC = "/ecocar/stereo/left/image_raw/compressed";
std::string CLOUD_TOPIC = "/ecocar/ouster/lidar_packets";

std::map<std::string, std::queue<sensor_msgs::CompressedImage::ConstPtr>> imageQueues;
std::queue<sensor_msgs::PointCloud2::ConstPtr> cloudQueue;

template<typename T>
void callback(const typename T::ConstPtr& msg, const std::string& topicName) {
    // Cast the queue back to the correct type (unsafe, for demonstration only)
    auto& q = imageQueues[topicName];
    q.push(msg);
    ROS_INFO("Message received on %s", topicName.c_str());
}

void imageCallback(const sensor_msgs::CompressedImage::ConstPtr& msg) {
    auto& q = imageQueues[IMAGE_TOPIC];
    q.push(msg);
    ROS_INFO("Message received on %s", IMAGE_TOPIC.c_str());
}

void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    auto& q = cloudQueue;
    q.push(msg);
    ROS_INFO("Message received on %s", CLOUD_TOPIC.c_str());
}


int main(int argc, char *argv[]) {
    if (argc < 3) {
        cout << "usage LaneDetection inputVideo outputVideo [--cuda|--seq]" << endl << endl;
        cout << "Positional Arguments:" << endl;
        cout << " inputVideo    Input video for which lanes are detected" << endl;
        cout << " outputVideo   Name of resulting output video" << endl << endl;
        cout << "Optional Arguments:" << endl;
        cout << " --cuda        Perform hough transform using CUDA (default)" << endl;
        cout << " --seq         Perform hough transform sequentially on the CPU" << endl;
        return 1;
    }

    YAML::Node settings = YAML::LoadFile("/root/cuda-lane-detection/config/leva.yaml");

    // ROS setup
    ros::init(argc, argv, "lane_detection");
    ros::NodeHandle nh;

    imageQueues[IMAGE_TOPIC] =  std::queue<sensor_msgs::CompressedImage::ConstPtr>();
    ros::Subscriber imageSub = nh.subscribe<sensor_msgs::CompressedImage>(IMAGE_TOPIC, 10, imageCallback);
    ros::Subscriber cloudSub = nh.subscribe<sensor_msgs::PointCloud2>(CLOUD_TOPIC, 10, cloudCallback);

    ros::Publisher  pub = nh.advertise<sensor_msgs::PointCloud2>("/leva/detected_lanes", 10);

    int houghStrategy = settings["houghStrategy"].as<std::string>() == "cuda" ? CUDA : SEQUENTIAL;
    int frameWidth = settings["frameWidth"].as<int>();
    int frameHeight = settings["frameHeight"].as<int>();

    VideoWriter video(argv[2], VideoWriter::fourcc('a', 'v', 'c', '1') , 10,
                      Size(frameWidth, frameHeight), true);

    // Load LiDAR Camera calibrations. 
    std::string lidar2cam0_fpath = "/robodata/ecocar_logs/processed/CACCDataset/calibrations/44/calib_os1_to_cam0.yaml";
    std::string lidar2cam1_fpath = "/robodata/ecocar_logs/processed/CACCDataset/calibrations/44/calib_os1_to_cam1.yaml";
    
    YAML::Node config_lidar2cam0 = YAML::LoadFile(lidar2cam0_fpath);
    YAML::Node config_lidar2cam1 = YAML::LoadFile(lidar2cam1_fpath);

    
    auto K_lidar2cam0_vec = config_lidar2cam0["extrinsic_matrix"]["data"].as<std::vector<double>>();
    auto K_lidar2cam1_vec = config_lidar2cam1["extrinsic_matrix"]["data"].as<std::vector<double>>();
    auto K_lidar2cam0 = vectorToMat(K_lidar2cam0_vec, 4, 4);
    auto K_lidar2cam1 = vectorToMat(K_lidar2cam1_vec, 4, 4);

    // Compute LiDAR to pixel matrix, Compute pixel to LiDAR Matrix
    cv::Mat T_lidar2pixel;
    cv::Mat T_pixel2lidar;

    HoughTransformHandle *handle;
    createHandle(handle, houghStrategy, frameWidth, frameHeight);
    while (ros::ok()) {
        for (auto& kv : imageQueues) {
            std::string topicName = kv.first;
            auto& imageQueue = kv.second;
            if (!imageQueue.empty() && !cloudQueue.empty()) {
                //1 Get timestamps for image and cloud
                ros::Time cloud_time = cloudQueue.front()->header.stamp;
                ros::Time image_time = imageQueue.front()->header.stamp;

                //2 Check if timestamps are within a certain threshold
                if (abs(cloud_time.toSec() - image_time.toSec()) > 0.1) {
                    ROS_WARN("Timestamps are not in sync. Skipping older frame");
                    ROS_WARN("Cloud: %f, Image: %f", cloud_time.toSec(), image_time.toSec());
                    if (cloud_time.toSec() > image_time.toSec()) {
                        imageQueue.pop();
                    } else {    
                        cloudQueue.pop();
                    }
                    continue;
                }

                std::cout << "Processing image from " << kv.first << "\n";
                ROS_INFO("Processing image from %s", topicName.c_str());
                sensor_msgs::CompressedImage::ConstPtr img_msg = imageQueue.front();
                sensor_msgs::PointCloud2::ConstPtr cloud_msg = cloudQueue.front();
                imageQueue.pop();
                cloudQueue.pop();
                
                //2 Detect lanes in image
                auto lines = detectLanes(img_msg, handle, video, houghStrategy);

                //3 Compute line anchor depths from lidar
                vector<LineAnchors> uvd;
                convert_lines_to_uvd(
                    lines, cloud_msg, T_lidar2pixel, frameHeight, frameWidth, uvd
                );

                //4 Backprojet 2d lines to 3d using camera calibrations + depth
                vector<LineAnchors> xyz;
                convert_uvd_to_xyz(uvd, T_pixel2lidar, xyz);
                
                //5 Convert lane detection to GM format [Ji-Hwan]
                // vector<cv::Point3f> xyz_GM = convert_world_to_gm(xyz)

                //5 Publish detected lanes over ROS [Ji-Hwan]
                // sensor_msgs::msg::PointCloud2 pc_msg;
                // // TODO: FILL pc_msg with xyz_GM
                // pc_msg  
                // pub.publish(pc_msg);

            }
        }
        ros::spinOnce();
    }

    destroyHandle(handle, houghStrategy);

    // // Read input video
    // VideoCapture capture(argv[1]);
    // Check which strategy to use for hough transform (CUDA or sequential)
    // int houghStrategy = argc > 3 && !strcmp(argv[3], "--seq") ? SEQUENTIAL : CUDA;
    // int frameWidth = capture.get(CAP_PROP_FRAME_WIDTH);
    // int frameHeight = capture.get(CAP_PROP_FRAME_HEIGHT);

    // if (!capture.isOpened()) {
    //     cout << "Unable to open video" << endl;
    //     return -1;
    // }

    // VideoWriter video(argv[2], VideoWriter::fourcc('a', 'v', 'c', '1') , 30,
    //                   Size(frameWidth, frameHeight), true);

    // detectLanes(capture, video, houghStrategy);

    return 0;
}

vector<Line> detectLanes(
    sensor_msgs::CompressedImage::ConstPtr msg, 
    HoughTransformHandle* handle, 
    VideoWriter& outputVideo,
    int houghStrategy
    ) {
    // Convert to OpenCV image
    cv::Mat frame = cv::imdecode(cv::Mat(msg->data), 1);

    cv::Mat preProcFrame;
    vector<Line> lines;

    preProcFrame = filterLanes(frame);
    preProcFrame = applyGaussianBlur(preProcFrame);
    preProcFrame = applyCannyEdgeDetection(preProcFrame);
    preProcFrame = regionOfInterest(preProcFrame);
    // cv::imwrite("roi.png", preProcFrame);

    lines.clear();
    if (houghStrategy == CUDA)
        houghTransformCuda(handle, preProcFrame, lines);
    else if (houghStrategy == SEQUENTIAL)
        houghTransformSeq(handle, preProcFrame, lines);

    drawLines(frame, lines);
    // cv::imwrite("frame.png", frame);
    outputVideo << frame;
    
    return lines;
}

/** Draws given lines onto frame */
void drawLines(Mat &frame, vector<Line> lines) {
    for (size_t i = 0; i < lines.size(); i++) {
        int y1 = frame.rows;
        int y2 = (frame.rows / 2) + (frame.rows / 10);
        int x1 = (int) lines[i].getX(y1);
        int x2 = (int) lines[i].getX(y2);

        line(frame, Point(x1, y1), Point(x2, y2), Scalar(255), 5, 8, 0);
    }
}

/**
 * Helper function which plots the accumulator and returns result image (only 
 * for debugging purposes)
 */
Mat plotAccumulator(int nRows, int nCols, int *accumulator) {
	Mat plotImg(nRows, nCols, CV_8UC1, Scalar(0));
	for (int i = 0; i < nRows; i++) {
  		for (int j = 0; j < nCols; j++) {
			plotImg.at<uchar>(i, j) = min(accumulator[(i * nCols) + j] * 4, 255);
  		}
  	}

    return plotImg;
}

/**
 * Convert lines to uvd coordinates in pixel space with depth information.
 *
 * @param lines Vector to which found lines are added to (N, 2)
 * @param lidar_fpath Path to the lidar data file
 * @param lidar2cam0_fpath Path to the lidar to camera extrinsic matrix
 * @param lidar2cam1_fpath Path to the lidar to camera extrinsic matrix
 * @return uvd Coordinates of the projection point in pixel space (N, 3)
 */
void convert_lines_to_uvd(
    const vector<Line>& lines, 
    const sensor_msgs::PointCloud2::ConstPtr cloud_msg, 
    const cv::Mat& T_lidar2pixel, 
    const size_t height,
    const size_t width,
    vector<LineAnchors> &uvd) {

    // pcl::PCLPointCloud2 pcl_pc2;
    // pcl_conversions::toPCL(*cloud_msg, pcl_pc2);
    // pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);

    // //0 Uniformly sample points along each line (u, v)

    // //1 Project lidar to image cooridnate
    // cv::Mat PC_pixel = T_lidar2pixel * PC_lidar

    // //2 Remove points outside of image


    // //3 Remove points behind image
    
    // //4 Get closest depth to each line anchor point 

    // //5 Backprojct line anchors to 3D

    // //6 Return L (# of lines) x n (# of anchors) x 3 (coordinates) vector

}

/**
 * REFERENCE — https://www.fdxlabs.com/calculate-x-y-z-real-world-coordinates-from-a-single-camera-using-opencv/
 * Project pixel space to world coordinate space.
 *
 * @param uvd Coordinates of the projection point in pixel space (N, 3)
 * @param T_cam_to_lid Homogeneous transformation matrix from camera to lidar (4, 4)
 * @return xyz - coordinates of a 3D point in the world coordinate space (N, 4)
 */
void convert_uvd_to_xyz(
    const vector<LineAnchors>& uvd,
    const cv::Mat& T_uvd2xyz,
    vector<LineAnchors>& xyz
) {
    // cv::Mat uvd_mat = convert_vec_to_mat(uvd).t();    // (4, N)
    // YAML::Node config = YAML::LoadFile("cam.yaml");
    // cv::Mat K = config["extrinsic_matrix"]["data"].as<cv::Mat>(); // (3, 4) Camera intrinsic matrix
    // cv::Mat C = cv::Mat::eye(4, 3, CV_32F);           // (4, 3) Canonical form of matrix

    // cv::Mat K_inv = K.inv();
    // cv::Mat xyz_mat = K_inv * uvd_mat;

    // vector<cv::Point3f> xyz = convert_mat_to_vec(xyz_mat.t()); // (N, 3)

    // return xyz
}

/**
 * REFERENCE https://learnopencv.com/rotation-matrix-to-euler-angles/
 * Calculates rotation matrix given euler angles. ZYX rotation order.
 *
 */
cv::Mat eulerAnglesToRotationMatrix(cv::Vec3f &theta) {
    // Calculate rotation about x axis
    cv::Mat R_x = (cv::Mat_<float>(3,3) <<
               1,       0,              0,
               0,       cos(theta[0]),   -sin(theta[0]),
               0,       sin(theta[0]),   cos(theta[0])
               );
 
    // Calculate rotation about y axis
    cv::Mat R_y = (cv::Mat_<float>(3,3) <<
               cos(theta[1]),    0,      sin(theta[1]),
               0,               1,      0,
               -sin(theta[1]),   0,      cos(theta[1])
               );
 
    // Calculate rotation about z axis
    cv::Mat R_z = (cv::Mat_<float>(3,3) <<
               cos(theta[2]),    -sin(theta[2]),      0,
               sin(theta[2]),    cos(theta[2]),       0,
               0,               0,                  1);
 
    // Combined rotation matrix
    cv::Mat R = R_z * R_y * R_x;
 
    return R;
}

cv::Mat buildHomogeneousMatrix(cv::Point3f trans, cv::Vec3f theta) {
    cv::Mat T = cv::Mat::eye(4, 4, CV_32F);
    
    // Translation
    T.at<float>(0, 3) = trans.x;
    T.at<float>(1, 3) = trans.y;
    T.at<float>(2, 3) = trans.z;

    // Rotation
    cv::Mat R = eulerAnglesToRotationMatrix(theta);
    
    R.copyTo(T(cv::Rect(0, 0, 3, 3)));    

    return T;
}

/*HELPER FUNCTIONS*/

// cv::Mat convert_vec_to_mat(const vector<cv::Point3f>& xyz) {
//     cv::Mat mat(xyz.size(), 1, CV_32FC3);
//     for (size_t i = 0; i < xyz.size(); i++) {
//         mat.at<cv::Vec3f>(i, 0) = cv::Vec3f(xyz[i].x, xyz[i].y, xyz[i].z, 1);
//     }
//     return mat; // (N, 4)
// }

// vector<cv::Point3f> convert_mat_to_vec(const cv::Mat& mat) {
//     vector<cv::Point3f> xyz;
//     for (int i = 0; i < mat.rows; i++) {
//         cv::Vec3f point = mat.at<cv::Vec3f>(i, 0);
//         xyz.push_back(cv::Point3f(point[0], point[1], point[2]));
//     }
//     return xyz; // (N, 3)
// }

/**
 * Project 3D points in world coordinate space to 3D points in GM coordinate space.
 *
 * @param xyz Coordinates of a 3D point of lanes in the world coordinate space (N, 4)
 * @return xyz_GM Coordinates of a 3D point of lanes in the GM coordinate space (N, 4)
 */
// vector<cv::Point3f> convert_world_to_gm(vector<cv::Point3f> xyz) {

//     cv::Mat xyz_mat = convert_vec_to_mat(xyz).t(); // (4, N)
    
//     cv::Point3f trans_XX = cv::Point3f(0, 0, 0);
//     cv::Vec3f theta_XX = cv::Vec3f(0, 0, 0);

//     cv::Point3f trans_YY = cv::Point3f(0, 0, 0);
//     cv::Vec3f theta_YY = cv::Vec3f(0, 0, 0);
    
//     cv::Point3f trans_ZZ = cv::Point3f(0, 0, 0);
//     cv::Vec3f theta_ZZ = cv::Vec3f(0, 0, 0);
    

//     cv::Mat T_XX = buildHomogeneousMatrix(trans_XX, theta_XX)
//     cv::Mat T_YY = buildHomogeneousMatrix(trans_YY, theta_YY)
//     cv::Mat T_ZZ = buildHomogeneousMatrix(trans_ZZ, theta_ZZ)

//     cv::Mat xyz_GM_mat = T_ZZ * T_YY * T_XX * xyz_mat;

//     vector<cv::Point3f> xyz_GM = convert_mat_to_vec(xyz_GM_mat.t()); // (N, 3)

//     return xyz_GM;
// }
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

#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/flann/flann.hpp>


extern void convert_lines_to_xyz(
    vector<Line>& lines,
    const sensor_msgs::PointCloud2::ConstPtr cloud_msg, 
    const cv::Mat& T_lidar2pixel, 
    const cv::Mat& T_pixel2lidar, 
    const size_t width,
    const size_t height,
    vector<cv::Mat> &xyz);
extern void convert_world_to_gm(vector<cv::Mat> xyz,
                                vector<cv::Mat>& xyz_GM);
extern vector<Line> detectLanes(sensor_msgs::CompressedImage::ConstPtr msg, 
                                HoughTransformHandle* handle, 
                                int houghStrategy);
extern void drawLines(Mat &frame, vector<Line> lines);
extern Mat plotAccumulator(int nRows, int nCols, int *accumulator);
extern cv::Mat vectorToMat(const std::vector<double>& vec, int rows, int cols) {
    cv::Mat mat(rows, cols, CV_32F);
    int index = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat.at<float>(i, j) = vec[index];
            index++;
        }
    }
    return mat;
};

// DEFINITION OF GLOBAL VARIABLES
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

/*HELPERS FUNCTION INITIALIZER*/
void samplePointsAlongLine(const cv::Point& startPoint, 
                           const cv::Point& endPoint, 
                           int numSamples,
                           cv::Mat& sampledPoints_mat);

void extractPoints(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg, 
                   cv::Mat& PC_lidar_mat);

void filterPoints(const cv::Mat& PC_pixel, 
                  const size_t width, 
                  const size_t height, 
                  cv::Mat& PC_pixel_filtered);

void getClosestDepth(const vector<cv::Mat>& sampledPoints_vec, 
                       const cv::Mat& PC_pixel,
                       vector<cv::Mat>& indexVector);

// int main_sub(int argc, char *argv[]) {
int main(int argc, char **argv) {
    // if (argc < 3) {
    //     cout << "usage LaneDetection inputVideo outputVideo [--cuda|--seq]" << endl << endl;
    //     cout << "Positional Arguments:" << endl;
    //     cout << " inputVideo    Input video for which lanes are detected" << endl;
    //     cout << " outputVideo   Name of resulting output video" << endl << endl;
    //     cout << "Optional Arguments:" << endl;
    //     cout << " --cuda        Perform hough transform using CUDA (default)" << endl;
    //     cout << " --seq         Perform hough transform sequentially on the CPU" << endl;
    //     return 1;
    // }

    YAML::Node settings = YAML::LoadFile("/root/cuda-lane-detection/config/leva.yaml");

    // ROS setup
    ros::init(argc, argv, "lane_detection");
    ros::NodeHandle nh;

    imageQueues[IMAGE_TOPIC] =  std::queue<sensor_msgs::CompressedImage::ConstPtr>();
    ros::Subscriber imageSub = nh.subscribe<sensor_msgs::CompressedImage>(IMAGE_TOPIC, 10, imageCallback);
    ros::Subscriber cloudSub = nh.subscribe<sensor_msgs::PointCloud2>(CLOUD_TOPIC, 10, cloudCallback);

    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("/leva/detected_lanes", 10);

    int houghStrategy = settings["houghStrategy"].as<std::string>() == "cuda" ? CUDA : SEQUENTIAL;
    int frameWidth = settings["frameWidth"].as<int>();
    int frameHeight = settings["frameHeight"].as<int>();

    YAML::Node lidar2cam0 = YAML::LoadFile("/root/cuda-lane-detection/calibrations/44/calib_os1_to_cam0.yaml");
    YAML::Node cam0_intrinsics = YAML::LoadFile("/root/cuda-lane-detection/calibrations/44/calib_cam0_intrinsics.yaml");

    auto T_lidar2cam0_vec = lidar2cam0["extrinsic_matrix"]["data"].as<std::vector<double>>();
    auto K_cam0_vec = cam0_intrinsics["camera_matrix"]["data"].as<std::vector<double>>();
    auto T_lidar2cam0 = vectorToMat(T_lidar2cam0_vec, 4, 4);
    auto K_cam0 = vectorToMat(K_cam0_vec, 3, 3);

    // REFERENCE https://www.fdxlabs.com/calculate-x-y-z-real-world-coordinates-from-a-single-camera-using-opencv/
    // multipling by the lidar to camera matrix, a 3x4 matrix with the 3x3 identity sublock on the left, and the camera intrinsics matrix
    // Compute LiDAR to pixel matrix, Compute pixel to LiDAR Matrix
    
    cv::Mat DUMMY = cv::Mat::eye(3, 4, CV_32F);
    cv::Mat T_lidar2pixel = K_cam0 * DUMMY * T_lidar2cam0; // (3,3) x (3, 4) x (4, 4) = (3, 4)

    DUMMY = cv::Mat::eye(4, 3, CV_32F);
    cv::Mat K_cam0_inv = K_cam0.inv();
    cv::Mat T_lidar2cam0_inv = T_lidar2cam0.inv();
    cv::Mat T_pixel2lidar = T_lidar2cam0_inv * DUMMY * K_cam0_inv; // (4,4) x (4, 3) x (3, 3) = (4, 3)

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
                auto lines = detectLanes(img_msg, handle, houghStrategy);

                //3 Compute line anchor depths from lidar & backproject 2d lines to 3d
                vector<cv::Mat> xyz;
                convert_lines_to_xyz(
                    lines, cloud_msg, T_lidar2pixel, T_pixel2lidar, frameHeight, frameWidth, xyz
                );
                
                //5 Convert lane detection to GM format [Ji-Hwan]
                vector<cv::Mat> xyz_GM;
                convert_world_to_gm(xyz, xyz_GM);
                
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
                for (const auto& line : xyz_GM) {
                    for (int i = 0; i < line.rows; i++) {
                        cloud->push_back(pcl::PointXYZ(line.at<float>(i, 0), line.at<float>(i, 1), line.at<float>(i, 2)));
                    }
                }

                //5 Publish detected lanes over ROS [Ji-Hwan]
                sensor_msgs::PointCloud2 pc_msg;
                pcl::toROSMsg(*cloud, pc_msg);
                pc_msg.header.stamp = ros::Time::now();
                pc_msg.header.frame_id = "lidar_frame";
                pub.publish(pc_msg);
            }
        }
        ros::spinOnce();
    }
    destroyHandle(handle, houghStrategy);

    return 0;
}

vector<Line> detectLanes(
    sensor_msgs::CompressedImage::ConstPtr msg, 
    HoughTransformHandle* handle, 
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
    // outputVideo << frame;
    
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

/*HELPER FUNCTIONS*/
void samplePointsAlongLine(const cv::Point& startPoint, 
                           const cv::Point& endPoint, 
                           int numSamples,
                           cv::Mat& sampledPoints_mat) {
    
    // Calculate the step size for sampling
    float stepSizeX = static_cast<float>(endPoint.x - startPoint.x) / (numSamples - 1);
    float stepSizeY = static_cast<float>(endPoint.y - startPoint.y) / (numSamples - 1);
    
    // Sample points along the line
    for (int i = 0; i < numSamples; i++) {
        int x = static_cast<int>(startPoint.x + i * stepSizeX);
        int y = static_cast<int>(startPoint.y + i * stepSizeY);
        cv::Mat pointMat = cv::Mat(1, 3, CV_32FC1);
        pointMat.at<float>(0, 0) = x;
        pointMat.at<float>(0, 1) = y;
        pointMat.at<float>(0, 2) = 0;
        sampledPoints_mat.push_back(pointMat);
    }
}

void extractPoints(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg, 
                   cv::Mat& PC_lidar_mat) {
    // https://docs.ros.org/en/indigo/api/pcl_conversions/html/pcl__conversions_8h_source.html#l00554
    // extract point cloud from message
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *temp_cloud);
    
    // convert point cloud to cv::Mat
    for (size_t i = 0; i < temp_cloud->height; i++) {
        for (size_t j = 0; j < temp_cloud->width; j++) {
            PC_lidar_mat.at<cv::Vec4f>(i * temp_cloud->width + j, 0) = cv::Vec4f(temp_cloud->points[i * temp_cloud->width + j].x,
                                                                                 temp_cloud->points[i * temp_cloud->width + j].y,
                                                                                 temp_cloud->points[i * temp_cloud->width + j].z,
                                                                                 1.0);
        }
    }
}

void filterPoints(const cv::Mat& PC_pixel, 
                  const size_t width, 
                  const size_t height, 
                  cv::Mat& PC_pixel_filtered) {

    for (int i = 0; i < PC_pixel.rows; i++) {
        // cv::Vec4f point = cv::Vec4f(PC_pixel.at<cv::Vec3f>(i, 0)[0], PC_pixel.at<cv::Vec3f>(i, 0)[1], PC_pixel.at<cv::Vec3f>(i, 0)[2], 1);
        float x = PC_pixel.at<cv::Vec3f>(i, 0)[0];
        float y = PC_pixel.at<cv::Vec3f>(i, 0)[1];
        float z = PC_pixel.at<cv::Vec3f>(i, 0)[2];

        // Remove points outside & behind of images
        if ((x > 0 && x <= width && y > 0 && y <= height && z > 0)) {
            cv::Mat pointMat = cv::Mat(1, 4, CV_32FC1);
            pointMat.at<float>(0, 0) = x;
            pointMat.at<float>(0, 1) = y;
            pointMat.at<float>(0, 2) = z;
            pointMat.at<float>(0, 3) = 1;
            PC_pixel_filtered.push_back(pointMat);
        }
    }
}

void getClosestDepth(const vector<cv::Mat>& sampledPoints_vec, 
                       const cv::Mat& PC_pixel,
                       vector<cv::Mat>& indexVector) {
    
    // create a copy with z-coordinate set to 0
    cv::Mat PC_pixel_copy = PC_pixel.colRange(0, 3).clone();
    PC_pixel_copy.col(2) = cv::Scalar(0);
    
    // Use kdtree to get index of the point cloud with the closest depth
    // Create a KD-Tree for the point cloud
    cv::flann::KDTreeIndexParams indexParams;
    cv::flann::Index kdtree(PC_pixel_copy, indexParams);

    for (const cv::Mat& sampledPoints_mat : sampledPoints_vec) {
        
        cv::Mat indices, dists;
        kdtree.knnSearch(sampledPoints_mat, indices, dists, 1);
        indexVector.push_back(indices);
    }
}

/**
 * Convert lines to uvd coordinates in pixel space with depth information.
 * Returns uvd with LineAnchors type (L (# of lines) x n (# of anchors) x 3 (coordinates) vector)
 *
 * @param lines vector to which found lines are added to (N, 2)
 * @param cloud_msg pointer to the point cloud message
 * @param T_lidar2pixel transformation matrix from lidar to pixel space (4, 4)
 * @param width width of the image
 * @param height height of the image
 * @return xyz coordinates of the projection point in pixel space with depth information (N, 3)
 */
void convert_lines_to_xyz(
    vector<Line>& lines,
    const sensor_msgs::PointCloud2::ConstPtr cloud_msg, 
    const cv::Mat& T_lidar2pixel, 
    const cv::Mat& T_pixel2lidar, 
    const size_t width,
    const size_t height,
    vector<cv::Mat> &xyz) {

    //0 Uniformly sample points along each line (u, v)
    // n # of Line object
    // for each line, we will have 10 anchor points
    vector<cv::Mat> sampledPoints_vec;
    size_t numAnchors = 10;

    for (size_t i = 0; i < lines.size(); i++) {
        cv::Mat sampledPoints_mat;
        // starting and ending points of a line
        int y1 = height;
        int y2 = (height / 2) + (height / 10);
        int x1 = (int) lines[i].getX(y1);
        int x2 = (int) lines[i].getX(y2);
        samplePointsAlongLine(Point(x1, y1), Point(x2, y2), numAnchors, sampledPoints_mat);
        sampledPoints_vec.push_back(sampledPoints_mat);
        }
    
    int numPoints = cloud_msg->width * cloud_msg->height;

    //1 Project lidar to (image) pixel cooridnates
    cv::Mat PC_lidar_mat(numPoints, 4, CV_32F);
    extractPoints(cloud_msg, PC_lidar_mat); // (N, 4)
    cv::Mat PC_pixel = (T_lidar2pixel * PC_lidar_mat.t()).t(); // (3, 4) x (4, N) = (3, N) - uvd coordinates in pixel space

    /* test if PC_lidar contains correct data */
    // pcl::io::savePCDFileASCII ("test_pcd.pcd", PC_lidar);
    // std::cerr << "Saved " << cloud.size () << " data points to test_pcd.pcd." << std::endl;
    
    //2 Remove points outside of images & behind images
    cv::Mat PC_pixel_filtered;
    filterPoints(PC_pixel, width, height, PC_pixel_filtered); // (N, 4)

    /* for debugging */
    // overlay points on image
    // cv::imwrite("overlay.png", frame)

    //4 Get closest depth (nearest search) to each line anchor point 
    vector<cv::Mat> indexVector;
    getClosestDepth(sampledPoints_vec, PC_pixel_filtered, indexVector);

    //5 Backproject line anchors to 3D (extract depth from projected lidar points and add to xyz)
    for (size_t i = 0; i < lines.size(); i++) {
        cv::Mat indices_line = indexVector[i];
        // LineAnchors lineAnchors(0);
        cv::Mat closestPoints;
        for (int j = 0; j < indices_line.rows; j++) {
            int index = indices_line.at<int>(j, 0);
            cv::Mat selectedPoint = PC_pixel_filtered.row(index);
            closestPoints.push_back(selectedPoint);
        }
        
        cv::Mat anchor_uvd_mat = sampledPoints_vec[i].clone();
        anchor_uvd_mat.col(1) = closestPoints.col(1);

        cv::Mat anchor_xyz_mat = (T_pixel2lidar * anchor_uvd_mat.t()).t(); // (4, 3) x (3, n) = (4, n). | (n, 4)
        xyz.push_back(anchor_xyz_mat);
    }
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
//         xyz.push_back(cv::Point3f(point[0], point[1], point[2]));convert_lines_to_uvd
//     }
//     return xyz; // (N, 3)
// }

/**
 * Project 3D points in world coordinate space to 3D points in GM coordinate space.
 *
 * @param xyz Coordinates of a 3D point of lanes in the world coordinate space (N, 4)
 * @return xyz_GM Coordinates of a 3D point of lanes in the GM coordinate space (N, 4)
 */
void convert_world_to_gm(vector<cv::Mat> xyz,
                         vector<cv::Mat>& xyz_GM) {

    cv::Point3f trans_lidar2GM = cv::Point3f(0, 0, -1.5);
    cv::Vec3f theta_lidar2GM = cv::Vec3f(0, 0, 0);
    cv::Mat T_lidar2GM = buildHomogeneousMatrix(trans_lidar2GM, theta_lidar2GM);

    for (size_t i = 0; i < xyz.size(); i++) {
        cv::Mat anchors_GM; // N number of anchors in xyz coord.
        anchors_GM = T_lidar2GM * xyz[i].t(); // (4, 4) x (4, N) = (4, N)
        xyz_GM.push_back(anchors_GM.t()); // L x (N, 4)
    }

    // for each line (xyz is a vector LineAnchors)
    // for (size_t i = 0; i < xyz.size(); i++) {
    //     cv::Mat anchors_GM; // N number of anchors in xyz coord.
    //     xyz[i].getAnchorsMat(anchors_GM); // (N, 4)
    //     anchors_GM = T_lidar2GM * anchors_GM.t(); // (4, 4) x (4, N) = (4, N)
    //     xyz_GM.push_back(anchors_GM.t()); // L x (N, 4)
    // }
}

// for (int j = 0; j < indices_line.rows; j++) {
    
//     int index = indices_line.at<int>(j, 0);
//     cv::Mat sampledPoints_mat = sampledPoints_vec[i];
//     cv::Point3f anchor_uvd(sampledPoints_mat.at<float>(j, 0),
//                            sampledPoints_mat.at<float>(j, 1),
//                            PC_pixel_filtered.at<float>(index, 2));
    
//     // Project pixel space to world coordinate space.
//     cv::Mat anchor_uvd_mat = (cv::Mat_<float>(1, 3) << anchor_uvd.x, anchor_uvd.y, anchor_uvd.z);
//     cv::Mat anchor_xyz_mat = T_pixel2lidar * anchor_uvd_mat; 
//     cv::Point3f anchor_xyz(anchor_xyz_mat.at<float>(0, 0),
//                            anchor_xyz_mat.at<float>(0, 1),
//                            anchor_xyz_mat.at<float>(0, 2));
//     lineAnchors.addAnchor(anchor_xyz);
// }
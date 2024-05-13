#include "commons.h"
#include "HoughTransform.h"
#include "Preprocessing.h"
#include <time.h>
#include <iomanip>
#include <queue>

#include <yaml-cpp/yaml.h>
#include <armadillo>

// ROS
#include "ros/ros.h"
#include "std_msgs/Header.h"
#include "std_msgs/Float32MultiArray.h"
#include "sensor_msgs/CompressedImage.h"
#include "sensor_msgs/PointCloud2.h"
#include <cv_bridge/cv_bridge.h>

// Packages
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/flann/flann.hpp>

extern void convert_lines_to_xyz(
    vector<Line> &lines,
    const sensor_msgs::CompressedImage::ConstPtr img_msg,
    const sensor_msgs::PointCloud2::ConstPtr cloud_msg,
    const cv::Mat &T_lidar_to_pixels,
    const cv::Mat &T_pixels_to_lidar,
    const size_t width,
    const size_t height,
    vector<cv::Mat> &xyz,
    cv::Mat &overlay_img);
extern void convert_world_to_gm(vector<cv::Mat> xyz,
                                vector<cv::Mat> &xyz_GM);
extern vector<Line> detectLanes(sensor_msgs::CompressedImage::ConstPtr msg,
                                HoughTransformHandle *handle,
                                int houghStrategy);
extern void drawLines(Mat &frame, vector<Line> lines);
extern Mat plotAccumulator(int nRows, int nCols, int *accumulator);
extern cv::Mat vectorToMat(const std::vector<double> &vec, int rows, int cols)
{
    cv::Mat mat(rows, cols, CV_32FC1);
    int index = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            mat.at<float>(i, j) = vec[index];
            index++;
        }
    }
    return mat;
};

// DEFINITION OF GLOBAL VARIABLES
std::string IMAGE_TOPIC = "/ecocar/stereo/left/image_raw/compressed";
std::string CLOUD_TOPIC = "/ecocar/ouster/lidar_packets";
std::string LANEDET_IMAGE_TOPIC = "/leva/lane_dt/stereo/left/image_raw/compressed";
std::string LANEDET_CLOUD_TOPIC = "/leva/lane_dt/ouster/lidar_packets";
std::string LANEDET_GM_CLOUD_TOPIC = "/leva/lane_dt/gm/pointclouds";
std::string LANEDET_GM_COEFF_TOPIC = "/leva/lane_dt/gm/coefficients";

std::map<std::string, std::queue<sensor_msgs::CompressedImage::ConstPtr>> imageQueues;
std::queue<sensor_msgs::PointCloud2::ConstPtr> cloudQueue;

template <typename T>
void callback(const typename T::ConstPtr &msg, const std::string &topicName)
{
    // Cast the queue back to the correct type (unsafe, for demonstration only)
    auto &q = imageQueues[topicName];
    q.push(msg);
    ROS_INFO("Message received on %s", topicName.c_str());
}

void imageCallback(const sensor_msgs::CompressedImage::ConstPtr &msg)
{
    auto &q = imageQueues[IMAGE_TOPIC];
    q.push(msg);
    ROS_INFO("Message received on %s", IMAGE_TOPIC.c_str());
}

void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    auto &q = cloudQueue;
    q.push(msg);
    ROS_INFO("Message received on %s", CLOUD_TOPIC.c_str());
}

/*HELPERS FUNCTION INITIALIZER*/
void samplePointsAlongLine(const cv::Point &startPoint,
                           const cv::Point &endPoint,
                           int numSamples,
                           cv::Mat &sampledPoints_mat);
void extractPoints(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
                   cv::Mat &PC_lidar_mat);
void pixels_to_depth(const cv::Mat &pc_np, const cv::Mat &T_lidar_to_pixels, int frameHeight, int frameWidth, cv::Mat &uvd);
void getClosestDepth(const vector<cv::Mat> &sampledPoints_vec,
                     const cv::Mat &PC_uvd,
                     vector<cv::Mat> &indexVector);
void saveDepthImage(const cv::Mat &depthMatrix, const std::string &filename);
void saveImage(cv::Mat image, vector<cv::Mat> sampledPoints_vec, cv::Mat PC_uvd_filtered, cv::Mat &overlay_img);
void extractCoeff(cv::Mat &xyz_best, arma::fvec &coefficients);
void cvMatToArmaMat(const cv::Mat &cvMat, arma::fmat &armaMat);
void create_pc_from_coeff(const std::vector<arma::fvec> &coefficients,
                          const std::vector<cv::Mat> &xyz_GM,
                          std::vector<cv::Mat> &xyz_GM_ld);
void armaMatToCvMat(const arma::fmat &armaMat, cv::Mat &cvMat);

void binLines(const std::vector<Line> &lines, 
    const std::vector<cv::Mat> &xyz_GM,
    std::vector<cv::Mat> &xyz_binned_GM,
    int laneWidth, 
    int numLanes, 
    int laneForwardMin, 
    int laneForwardMax
);

// void binLines(const vector<cv::Mat>& xyz_GM, int y_min, int y_max, int X_MIN, cv::Mat& xyz_best);

int main(int argc, char **argv)
{

    cout << "Started lane detection node" << endl;

    YAML::Node settings = YAML::LoadFile("/root/cuda-lane-detection/config/leva.yaml");
    // TODO: Enabled reading arbitrary number of image topics from config file

    // ROS setup
    ros::init(argc, argv, "lane_detection");
    ros::NodeHandle nh;

    imageQueues[IMAGE_TOPIC] = std::queue<sensor_msgs::CompressedImage::ConstPtr>();
    ros::Subscriber imageSub = nh.subscribe<sensor_msgs::CompressedImage>(IMAGE_TOPIC, 10, imageCallback);
    ros::Subscriber cloudSub = nh.subscribe<sensor_msgs::PointCloud2>(CLOUD_TOPIC, 10, cloudCallback);

    ros::Publisher lanedet_image_pub = nh.advertise<sensor_msgs::CompressedImage>(LANEDET_IMAGE_TOPIC, 10);
    ros::Publisher lanedet_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>(LANEDET_CLOUD_TOPIC, 10);
    ros::Publisher lanedet_coeff_pub = nh.advertise<std_msgs::Float32MultiArray>(LANEDET_GM_COEFF_TOPIC, 10);
    ros::Publisher lanedet_cloud_gm_pub = nh.advertise<sensor_msgs::PointCloud2>(LANEDET_GM_CLOUD_TOPIC, 10);

    int houghStrategy = settings["houghStrategy"].as<std::string>() == "cuda" ? CUDA : SEQUENTIAL;
    int frameWidth = settings["frameWidth"].as<int>();
    int frameHeight = settings["frameHeight"].as<int>();
    int laneWidth = settings["laneWidth"].as<int>();
    int numLanes = settings["numLanes"].as<int>();
    int laneForwardMin = settings["laneForwardMin"].as<int>();
    int laneForwardMax = settings["laneForwardMax"].as<int>();
    const int debugMode = settings["debugMode"].as<int>();

    YAML::Node lidar2cam0 = YAML::LoadFile("/root/cuda-lane-detection/calibrations/44/calib_os1_to_cam0.yaml");
    YAML::Node cam0_intrinsics = YAML::LoadFile("/root/cuda-lane-detection/calibrations/44/calib_cam0_intrinsics.yaml");

    auto lidar2cam0_vec = lidar2cam0["extrinsic_matrix"]["data"].as<std::vector<double>>();
    auto K_vec = cam0_intrinsics["camera_matrix"]["data"].as<std::vector<double>>();
    auto T_lidar_to_cam = vectorToMat(lidar2cam0_vec, 4, 4);
    auto K = vectorToMat(K_vec, 3, 3);

    cv::Mat T_canon = cv::Mat::eye(3, 4, CV_32FC1);
    cv::Mat T_lidar_to_pixels = K * T_canon * T_lidar_to_cam; // (3,3) x (3, 4) x (4, 4) = (3, 4)

    cv::Mat K_inv = K.inv();
    cv::Mat T_canon_inv = cv::Mat::eye(4, 3, CV_32FC1);
    cv::Mat T_lidar_to_cam_inv = T_lidar_to_cam.inv();
    cv::Mat T_pixels_to_lidar = T_lidar_to_cam_inv * T_canon_inv * K_inv; // (4,4) x (4, 3) x (3, 3) = (4, 3)
    std::cout << "Finished initializing config files" << std::endl;
    cv_bridge::CvImage img_bridge;
    sensor_msgs::CompressedImage compressed_img_msg; // >> message to be sent
    HoughTransformHandle *handle;
    createHandle(handle, houghStrategy, frameWidth, frameHeight);
    std::cout << "Finished initializing cuda handles " << std::endl;
    while (ros::ok())
    {
        for (auto &kv : imageQueues)
        {
            std::string topicName = kv.first;
            auto &imageQueue = kv.second;
            if (!imageQueue.empty() && !cloudQueue.empty())
            {
                // 1 Get timestamps for image and cloud
                ros::Time cloud_time = cloudQueue.front()->header.stamp;
                ros::Time image_time = imageQueue.front()->header.stamp;

                // 2 Check if timestamps are within a certain threshold
                if (abs(cloud_time.toSec() - image_time.toSec()) > 0.1)
                {
                    ROS_WARN("Timestamps are not in sync. Skipping older frame");
                    ROS_WARN("Cloud: %f, Image: %f", cloud_time.toSec(), image_time.toSec());
                    if (cloud_time.toSec() > image_time.toSec())
                    {
                        imageQueue.pop();
                    }
                    else
                    {
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

                // 2 Detect lanes in image
                auto lines = detectLanes(img_msg, handle, houghStrategy);
                std::cout << "Detected lanes " << "\n";

                // 3 Compute line anchor depths from lidar & backproject 2d lines to 3d
                vector<cv::Mat> xyz;
                cv::Mat overlay_img;
                convert_lines_to_xyz(
                    lines, img_msg, cloud_msg, T_lidar_to_pixels, T_pixels_to_lidar, frameWidth, frameHeight, xyz, overlay_img
                );

                // 5 Convert lane detection to GM format [Ji-Hwan]ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
                vector<cv::Mat> xyz_GM;
                convert_world_to_gm(xyz, xyz_GM);

                std::vector<cv::Mat> xyz_binned_GM(6, cv::Mat::zeros(12, 3, CV_32FC1));
                binLines(
                    lines,
                    xyz_GM,
                    xyz_binned_GM,
                    laneWidth,
                    numLanes,
                    laneForwardMin,
                    laneForwardMax
                );

                if (debugMode==2) {
                    // 6 Print xyz_GM in terminal
                    for (const auto& line : xyz_binned_GM) {
                        std::cout << "xyz_binned_GM line:" << std::endl;
                        for (int i = 0; i < line.rows; i++) {
                            std::cout << line.at<float>(i, 0) << ", " << line.at<float>(i, 1) << ", " << line.at<float>(i, 2) << std::endl;
                            if (i > 5) break;
                        }
                    }
                    for (const auto& line : xyz_GM) {
                        std::cout << "xyz_GM line:" << std::endl;
                        for (int i = 0; i < line.rows; i++) {
                            std::cout << line.at<float>(i, 0) << ", " << line.at<float>(i, 1) << ", " << line.at<float>(i, 2) << std::endl;
                            if (i > 5) break;
                        }
                    }
                }

                //7 Compute coefficients for each line
                vector<arma::fvec> coefficients_vec;
                for (auto& xyz_mat : xyz_binned_GM) {
                    arma::fvec coefficients(4, arma::fill::zeros);
                    extractCoeff(xyz_mat, coefficients);
                    coefficients_vec.push_back(coefficients);
                }

                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
                for (const auto &line : xyz_binned_GM)
                {
                    for (int i = 0; i < line.rows; i++)
                    {
                        cloud->push_back(pcl::PointXYZ(line.at<float>(i, 0), line.at<float>(i, 1), line.at<float>(i, 2)));
                    }
                }

                // 5 Publish detected lanes over ROS
                sensor_msgs::PointCloud2 pc_msg;
                pcl::toROSMsg(*cloud, pc_msg);
                pc_msg.header.seq = cloud_msg->header.seq;
                pc_msg.header.stamp = cloud_time;
                pc_msg.header.frame_id = "os_sensor";
                lanedet_cloud_pub.publish(pc_msg);

                // PUBLISH GM LANE COEFFICIENTS
                std_msgs::Float32MultiArray coeff_msg;
                coeff_msg.data.clear();
                for (arma::fvec value : coefficients_vec)
                {
                    std::vector<float> vec(value.begin(), value.end());
                    coeff_msg.data.insert(coeff_msg.data.end(), vec.begin(), vec.end());
                }
                lanedet_coeff_pub.publish(coeff_msg);

                if (debugMode==1) {
                    // POINTCLOUD VISUALIZATION USING GM LANE COEFFICIENTS
                    vector<cv::Mat> xyz_GM_ld;
                    create_pc_from_coeff(coefficients_vec, xyz_binned_GM, xyz_GM_ld);

                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_gm_ld(new pcl::PointCloud<pcl::PointXYZ>);
                    for (const auto &line : xyz_GM_ld)
                    {
                        for (int i = 0; i < line.rows; i++)
                        {
                            cloud_gm_ld->push_back(pcl::PointXYZ(line.at<float>(i, 0), line.at<float>(i, 1), line.at<float>(i, 2)));
                        }
                    }
                    sensor_msgs::PointCloud2 pc_gm_msg;
                    pcl::toROSMsg(*cloud_gm_ld, pc_gm_msg);
                    pc_gm_msg.header.seq = cloud_msg->header.seq;
                    pc_gm_msg.header.stamp = cloud_time;
                    pc_gm_msg.header.frame_id = "os_sensor";
                    lanedet_cloud_gm_pub.publish(pc_gm_msg);
                }

                if (debugMode==1) {
                    // Add overlay_img to img_msg
                    std_msgs::Header header;
                    header.seq = img_msg->header.seq;
                    header.stamp = cloud_time;
                    header.frame_id = "os_sensor";
                    img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, overlay_img);
                    img_bridge.toCompressedImageMsg(compressed_img_msg); // from cv_bridge to sensor_msgs::CompressedImage
                    lanedet_image_pub.publish(compressed_img_msg);
                }
            }
        }
        ros::spinOnce();
    }
    destroyHandle(handle, houghStrategy);

    return 0;
}

/*
This function bins each may in xyz_GM based that are within the x range
and y range specified by the parameters. The line with the most points (accum)
for each bin is returned for each bin.
*/
void binLines(const std::vector<Line> &lines, 
    const std::vector<cv::Mat> &xyz_GM,
    std::vector<cv::Mat> &xyz_binned_GM,
    int laneWidth, 
    int numLanes, 
    int laneForwardMin, 
    int laneForwardMax
) {
    std::unordered_map<int, std::vector<int> > binnedLines;
    //1 For each xyz_GM point, count the number of points in each bin
    // and store the bin with the most points for each line
    float lpm = 1.0f / laneWidth; // lanes per meter
    float laneMin = -11; // manually tuned from lidar
    for (size_t lineIdx = 0; lineIdx < lines.size(); lineIdx++) {
        const auto& ptsMat = xyz_GM[lineIdx];

        std::vector<int> binCounts(numLanes, 0);
        for (int i = 0; i < ptsMat.rows; i++) {
            float x = ptsMat.at<float>(i, 0);
            float y = ptsMat.at<float>(i, 1);
            if (x > laneForwardMin && x < laneForwardMax) {
                int laneBin = (int)floor((y - laneMin) * lpm);
                if (laneBin >= 0 && laneBin < numLanes) {
                    binCounts[laneBin]++;
                }
            }
        }

        // Assign the line to the lane with the most points
        int bestLane = -1;
        for (int i = 0; i < (int)binCounts.size(); ++i) {
            if (binCounts[i]==0) continue;

            if (bestLane==-1) {
                bestLane = i;
            } else if (binCounts[i] >  binCounts[bestLane]) {
                bestLane = i;
            }
        }
        if (bestLane>-1) {
            binnedLines[bestLane].push_back(lineIdx);
        }
    }

    //2 For each lane, select the cv::Mat that corresponds
    // to the line with the most points
    std::vector<int> laneToLineId(numLanes, -1);
    for (const auto&kv : binnedLines) {
        auto laneId = kv.first;
        const auto& lineIds = kv.second;

        int bestLineId = -1;
        for (auto lineId : lineIds) {
            if (bestLineId==-1) {
                bestLineId = lineId;
            } else if (lines[lineId].getAccum() > lines[bestLineId].getAccum()) {
                bestLineId = lineId;
            }
        }
        laneToLineId[laneId] = bestLineId;
    }

    //3 Select the best xyz_GM cv::Mat for each lane
    std::vector<int> gmLaneOrder = {4, 2, 0, 1, 3, 5};
    for (size_t i = 0; i < laneToLineId.size(); ++i) {
        if (laneToLineId[i] > -1) {
            xyz_binned_GM[gmLaneOrder[i]] = xyz_GM[laneToLineId[i]];
        } else {
            xyz_binned_GM[gmLaneOrder[i]] = cv::Mat::zeros(12, 3, CV_32FC1); // (N, 3)
        }
    }
}

// void binLines(const vector<cv::Mat>& xyz_GM, int y_min, int y_max, int X_MIN, cv::Mat& xyz_best) {

//     int cnt_best = std::numeric_limits<int>::min();
//     for (const auto& line : xyz_GM) {
//         int count = 0;
//         for (int i = 0; i < line.rows; i++) {
//             cv::Mat point = line.row(i);
//             float x = point.at<float>(0);
//             float y = point.at<float>(1);
//             if (y >= y_min && y <= y_max && x >= X_MIN) {
//                 count++;
//             }
//         }
//         if (count != 0 && count > cnt_best) {
//             cnt_best = count;
//             std::cout << cnt_best << std::endl;
//             xyz_best = line.clone();
//         }
//     }
// }

void create_pc_from_coeff(const std::vector<arma::fvec> &coefficients,
                          const std::vector<cv::Mat> &xyz_GM,
                          std::vector<cv::Mat> &xyz_GM_ld) {
    for (size_t i = 0; i < xyz_GM.size(); i++)
    {
        const cv::Mat &mat = xyz_GM[i];
        cv::Mat xyz_GM_ld_i = mat.clone();
        const arma::fvec &coefficient = coefficients[i];

        cv::Mat x = mat.col(0);
        arma::fmat arma_x;
        cvMatToArmaMat(x, arma_x);

        // Prepare the design matrix A for cubic polynomial fitting
        arma::fmat A(arma_x.n_elem, 4);
        A.col(0) = arma::pow(arma_x, 3);
        A.col(1) = arma::pow(arma_x, 2);
        A.col(2) = arma_x;
        A.col(3).ones();

        arma::fmat arma_y = A * arma::conv_to<arma::fmat>::from(coefficient); // (n, 4) x (4, 1) = (n, 1)
        cv::Mat cv_y(arma_y.n_rows, arma_y.n_cols, CV_32FC1);
        armaMatToCvMat(arma_y, cv_y);

        cv_y.copyTo(xyz_GM_ld_i.col(1));
        // cout << xyz_GM_ld_i << endl;
        xyz_GM_ld.push_back(xyz_GM_ld_i);
    }
}

vector<Line> detectLanes(
    sensor_msgs::CompressedImage::ConstPtr msg,
    HoughTransformHandle *handle,
    int houghStrategy)
{
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
    std::cout << "Performed hough transform\n";
    drawLines(frame, lines);
    // cv::imwrite("frame.png", frame);
    // outputVideo << frame;

    return lines;
}

/** Draws given lines onto frame */
void drawLines(Mat &frame, vector<Line> lines)
{
    for (size_t i = 0; i < lines.size(); i++)
    {
        int y1 = frame.rows;
        int y2 = (frame.rows / 2) + (frame.rows / 10);
        int x1 = (int)lines[i].getX(y1);
        int x2 = (int)lines[i].getX(y2);

        line(frame, Point(x1, y1), Point(x2, y2), Scalar(255), 5, 8, 0);
    }
}

/**
 * Helper function which plots the accumulator and returns result image (only
 * for debugging purposes)
 */
Mat plotAccumulator(int nRows, int nCols, int *accumulator)
{
    Mat plotImg(nRows, nCols, CV_8UC1, Scalar(0));
    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nCols; j++)
        {
            plotImg.at<uchar>(i, j) = min(accumulator[(i * nCols) + j] * 4, 255);
        }
    }

    return plotImg;
}

/*HELPER FUNCTIONS*/
void samplePointsAlongLine(const cv::Point &startPoint,
                           const cv::Point &endPoint,
                           int numSamples,
                           cv::Mat &sampledPoints_mat)
{

    // Calculate the step size for sampling
    float stepSizeX = static_cast<float>(endPoint.x - startPoint.x) / (numSamples - 1);
    float stepSizeY = static_cast<float>(endPoint.y - startPoint.y) / (numSamples - 1);

    // Sample points along the line
    for (int i = 0; i < numSamples; i++)
    {
        // if (i > numSamples / 2) {
        // }
        int u = static_cast<int>(startPoint.x + i * stepSizeX);
        int v = static_cast<int>(startPoint.y + i * stepSizeY);
        cv::Mat pointMat = cv::Mat(1, 3, CV_32FC1);
        pointMat.at<float>(0, 0) = u;
        pointMat.at<float>(0, 1) = v;
        pointMat.at<float>(0, 2) = 1;
        sampledPoints_mat.push_back(pointMat);
    }
}

void extractPoints(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,
                   cv::Mat &PC_lidar_mat)
{

    // extract point cloud from message
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *temp_cloud); // temp_cloud - (heigh, width) of (x, y, z)

    int height = temp_cloud->height;
    int width = temp_cloud->width;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {

            float x = temp_cloud->at(j, i).x;
            float y = temp_cloud->at(j, i).y;
            float z = temp_cloud->at(j, i).z;

            if (x > -10 && x < 50 && z < 0)
            {
                cv::Mat pointMat = cv::Mat(1, 3, CV_32FC1);
                pointMat.at<float>(0, 0) = x;
                pointMat.at<float>(0, 1) = y;
                pointMat.at<float>(0, 2) = z;
                PC_lidar_mat.push_back(pointMat);
            }
        }
    }
}

void getClosestDepth(const vector<cv::Mat> &sampledPoints_vec,
                     const cv::Mat &PC_uvd,
                     vector<cv::Mat> &indexVector)
{

    // create a copy with z-coordinate set to 0
    cv::Mat uv1 = PC_uvd.clone();
    cv::Mat d = PC_uvd.col(PC_uvd.cols - 1);
    for (int i = 0; i < uv1.cols - 1; i++)
    {
        uv1.col(i) /= d; // Perspective division - [u v 1] d / d --> (u, v, 1)
    }
    uv1.col(2) = cv::Scalar(1);

    // Use kdtree to get index of the point cloud with the closest depth
    // Create a KD-Tree for the point cloud
    cv::flann::KDTreeIndexParams indexParams;
    cv::flann::Index kdtree(uv1, indexParams);

    for (const cv::Mat &sampledPoints_mat : sampledPoints_vec)
    {
        cv::Mat indices, dists;
        kdtree.knnSearch(sampledPoints_mat, indices, dists, 1);
        indexVector.push_back(indices);
    }
}

/**
 * Convert pixel coordinates to depth coordinates using the point cloud.
 *
 * @param lines vector to which found lines are added to (N, 2)
 * @return valid_points (N, 2)
 * @return valid_depths (N, 1)
 */
void pixels_to_depth(const cv::Mat &pc_np, const cv::Mat &T_lidar_to_pixels, int frameHeight, int frameWidth, cv::Mat &uvd)
{

    // Assume pc_np is a N x 3 matrix for point cloud
    cv::Mat pc_ones = cv::Mat::ones(pc_np.rows, 1, CV_32FC1); // (N, 1)
    cv::Mat pc_homo;
    cv::hconcat(pc_np, pc_ones, pc_homo); // pc_homo (N, 4)

    cv::Mat uv1d = T_lidar_to_pixels * pc_homo.t(); // (3, 4) x (4, N) = (3, N)
    uv1d = uv1d.t();                                // (N, 3) uvd

    // Remove points behind camera after coordinate system change
    cv::Mat uv1 = uv1d.clone();
    cv::Mat d = uv1d.col(uv1d.cols - 1);
    for (int i = 0; i < uv1.cols - 1; i++)
    {
        uv1.col(i) /= d; // Perspective division - [u v 1] d / d --> (u, v, 1)
    }

    for (int i = 0; i < uv1d.rows; i++)
    {
        float u = uv1.at<float>(i, 0);
        float v = uv1.at<float>(i, 1);
        float d = uv1d.at<float>(i, 2);

        // Remove points outside & behind of images
        if ((u >= 0 && u < frameWidth && v >= 0 && v < frameHeight && d > 1.0))
        {
            // copy over uv1d to pointMat
            cv::Mat pointMat = (cv::Mat_<float>(1, 3) << uv1d.at<float>(i, 0), uv1d.at<float>(i, 1), uv1d.at<float>(i, 2));
            uvd.push_back(pointMat);
        }
    }
}

void saveDepthImage(const cv::Mat &depthMatrix, const std::string &filename)
{
    // Assuming depthMatrix is of size Nx3 where N is the number of points.
    // Each row: [u, v, depth]
    if (depthMatrix.empty() || depthMatrix.cols != 3)
    {
        std::cerr << "Invalid input matrix." << std::endl;
        return;
    }

    int maxU = 960, maxV = 600;

    // Create an empty image with the same dimensions as the input matrix.
    // Initialize all pixels to black (depth = 0)
    cv::Mat depthImage = cv::Mat::zeros(maxV, maxU, CV_8UC1);
    // Find min and max depth values to scale depth to 0-255
    double minDepth, maxDepth;
    cv::minMaxIdx(depthMatrix.col(2), &minDepth, &maxDepth);
    // Iterate over the matrix to set pixel values in the depth image
    for (int i = 0; i < depthMatrix.rows; i++)
    {
        int u = static_cast<int>(depthMatrix.at<float>(i, 0));
        int v = static_cast<int>(depthMatrix.at<float>(i, 1));
        float depthValue = depthMatrix.at<float>(i, 2);
        // Normalize depth to 0-255 range for visualization
        uchar normalizedDepth = static_cast<uchar>(255 * (depthValue - minDepth) / (maxDepth - minDepth));
        // std::cout << "u: " << u << ", v: " << v << ", normalizedDepth: " << depthValue << std::endl;

        depthImage.at<uchar>(u, v) = normalizedDepth;
    }
    // Save the depth image
    if (!cv::imwrite(filename, depthImage))
    {
        std::cerr << "Failed to save the depth image" << std::endl;
    }
    else
    {
        std::cout << "Depth image saved successfully." << std::endl;
    }
}

void saveImage(cv::Mat image, vector<cv::Mat> sampledPoints_vec, cv::Mat PC_uvd_filtered, cv::Mat &overlay_img)
{

    // create a copy with z-coordinate set to 0
    cv::Mat uv1 = PC_uvd_filtered.clone();
    cv::Mat d = PC_uvd_filtered.col(PC_uvd_filtered.cols - 1);
    for (int i = 0; i < uv1.cols - 1; i++)
    {
        uv1.col(i) /= d; // Perspective division - [u v 1] d / d --> (u, v, 1)
    }
    uv1.col(2) = cv::Scalar(1);

    // Overlay red circles for uv1
    for (int i = 0; i < uv1.rows; i++)
    {
        int u = static_cast<int>(uv1.at<float>(i, 0));
        int v = static_cast<int>(uv1.at<float>(i, 1));
        cv::circle(image, cv::Point(u, v), 3, cv::Scalar(0, 0, 255), -1);
    }

    // Overlay green circles for sampledPoints_vec
    for (const cv::Mat &sampledPoints : sampledPoints_vec)
    {
        for (int i = 0; i < sampledPoints.rows; i++)
        {
            int u = static_cast<int>(sampledPoints.at<float>(i, 0));
            int v = static_cast<int>(sampledPoints.at<float>(i, 1));
            cv::circle(image, cv::Point(u, v), 5, cv::Scalar(0, 255, 0), -1);
        }
    }
    overlay_img = image;
}

void cvMatToArmaMat(const cv::Mat &cvMat, arma::fmat &armaMat)
{

    armaMat.set_size(cvMat.rows, cvMat.cols);
    for (int i = 0; i < cvMat.rows; i++)
    {
        for (int j = 0; j < cvMat.cols; j++)
        {
            armaMat(i, j) = cvMat.at<float>(i, j);
        }
    }
    // armaMat = arma::fmat(reinterpret_cast<float*>(cvMat.data), cvMat.rows, cvMat.cols);
}

void armaMatToCvMat(const arma::fmat &armaMat, cv::Mat &cvMat)
{
    std::memcpy(cvMat.data, armaMat.memptr(), armaMat.n_elem * sizeof(float));
}

void extractCoeff(cv::Mat &xyz_best, arma::fvec &coefficients)
{
    // Extract x and y columns directly
    cv::Mat x = xyz_best.col(0);
    cv::Mat y = xyz_best.col(1);

    // Convert OpenCV Mats to Armadillo matrices
    arma::fmat arma_x, arma_y;
    cvMatToArmaMat(x, arma_x);
    cvMatToArmaMat(y, arma_y);

    // Prepare the design matrix A for cubic polynomial fitting
    arma::fmat A(arma_x.n_elem, 4);
    A.col(0) = arma::pow(arma_x, 3);
    A.col(1) = arma::pow(arma_x, 2);
    A.col(2) = arma_x;
    A.col(3).ones();

    coefficients = arma::solve(A, arma_y);
}

/**
 * Convert lines to uvd coordinates in pixel space with depth information.
 * Returns uvd with LineAnchors type (L (# of lines) x n (# of anchors) x 3 (coordinates) vector)
 *
 * @param lines vector to which found lines are added to (N, 2)
 * @param cloud_msg pointer to the point cloud message
 * @param T_lidar_to_pixels transformation matrix from lidar to pixel space (4, 4)
 * @param width width of the image
 * @param height height of the image
 * @return xyz coordinates of the projection point in pixel space with depth information (N, 3)
 */
void convert_lines_to_xyz(
    vector<Line> &lines,
    const sensor_msgs::CompressedImage::ConstPtr img_msg,
    const sensor_msgs::PointCloud2::ConstPtr cloud_msg,
    const cv::Mat &T_lidar_to_pixels,
    const cv::Mat &T_pixels_to_lidar,
    const size_t width,
    const size_t height,
    vector<cv::Mat> &xyz,
    cv::Mat &overlay_img)
{

    // Convert to OpenCV image
    cv::Mat img = cv::imdecode(cv::Mat(img_msg->data), 1);

    // 0 Uniformly sample points along each line (u, v)
    //  n # of Line object. for each line, numAnchors anchor points
    vector<cv::Mat> sampledPoints_vec;
    size_t numAnchors = 20;
    for (size_t i = 0; i < lines.size(); i++)
    {
        cv::Mat sampledPoints_mat;
        // starting and ending points of a line
        // int y1 = height;
        int y1 = 0.75 * height; // projected pcs starting from 3/4 of the image
        int y2 = (height / 2) + (height / 10);
        int x1 = (int)lines[i].getX(y1);
        int x2 = (int)lines[i].getX(y2);

        samplePointsAlongLine(Point(x1, y1), Point(x2, y2), numAnchors, sampledPoints_mat);
        sampledPoints_vec.push_back(sampledPoints_mat);
    }

    // 1 Project lidar to (image) pixel cooridnates | TESTED - CONFRIMED
    cv::Mat PC_lidar_mat;
    extractPoints(cloud_msg, PC_lidar_mat); // (N, 3)

    // TESTED - CONFRIMED | PC_uvd_filtered looks good frin saveImage()
    cv::Mat PC_uv1d_filtered; // (u, v, d)
    pixels_to_depth(PC_lidar_mat, T_lidar_to_pixels, height, width, PC_uv1d_filtered);

    saveImage(img, sampledPoints_vec, PC_uv1d_filtered, overlay_img); // CONFIRMED
    // saveDepthImage(PC_uv1d_filtered, "depth.png");

    // 3 Get closest depth (nearest search) to each line anchor point
    vector<cv::Mat> indexVector;
    getClosestDepth(sampledPoints_vec, PC_uv1d_filtered, indexVector);

    std::cout << "backprojecting ... " << std::endl;

    // 4 Backproject line anchors to 3D (extract depth from projected lidar points and add to xyz)
    for (size_t i = 0; i < lines.size(); i++)
    {
        cv::Mat indices_line = indexVector[i];
        cv::Mat closestPoints;
        for (int j = 0; j < indices_line.rows; j++)
        {
            int index = indices_line.at<int>(j, 0);
            cv::Mat selectedPoint = PC_uv1d_filtered.row(index);
            closestPoints.push_back(selectedPoint);
        }
        cv::Mat anchor_uvd_mat = sampledPoints_vec[i].clone(); // (n, 3) uv1

        // Multiply each column of anchor_uvd_mat with the second column of closestPoints
        for (int col = 0; col < anchor_uvd_mat.cols; col++)
        {
            anchor_uvd_mat.col(col) = anchor_uvd_mat.col(col).mul(closestPoints.col(2)); // uv1 * uv1d
        }

        cv::Mat anchor_xyz_mat = (T_pixels_to_lidar * anchor_uvd_mat.t()).t(); // (4, 3) x (3, n) = (4, n). | (n, 4) xyz0
        anchor_xyz_mat = anchor_xyz_mat.colRange(0, anchor_xyz_mat.cols - 1);  // (uvd -> xyz)
        xyz.push_back(anchor_xyz_mat);
    }
}

/**
 * REFERENCE https://learnopencv.com/rotation-matrix-to-euler-angles/
 * Calculates rotation matrix given euler angles. ZYX rotation order.
 *
 */
cv::Mat eulerAnglesToRotationMatrix(cv::Vec3f &theta)
{
    // Calculate rotation about x axis
    cv::Mat R_x = (cv::Mat_<float>(3, 3) << 1, 0, 0,
                   0, cos(theta[0]), -sin(theta[0]),
                   0, sin(theta[0]), cos(theta[0]));

    // Calculate rotation about y axis
    cv::Mat R_y = (cv::Mat_<float>(3, 3) << cos(theta[1]), 0, sin(theta[1]),
                   0, 1, 0,
                   -sin(theta[1]), 0, cos(theta[1]));

    // Calculate rotation about z axis
    cv::Mat R_z = (cv::Mat_<float>(3, 3) << cos(theta[2]), -sin(theta[2]), 0,
                   sin(theta[2]), cos(theta[2]), 0,
                   0, 0, 1);

    // Combined rotation matrix
    cv::Mat R = R_z * R_y * R_x;

    return R;
}

cv::Mat buildHomogeneousMatrix(cv::Point3f trans, cv::Vec3f theta)
{
    cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);

    // Translation
    T.at<float>(0, 3) = trans.x;
    T.at<float>(1, 3) = trans.y;
    T.at<float>(2, 3) = trans.z;

    // Rotation
    cv::Mat R = eulerAnglesToRotationMatrix(theta);

    R.copyTo(T(cv::Rect(0, 0, 3, 3)));

    return T;
}

/**
 * Project 3D points in world coordinate space to 3D points in GM coordinate space.
 *
 * @param xyz Coordinates of a 3D point of lanes in the world coordinate space (N, 3)
 * @return xyz_GM Coordinates of a 3D point of lanes in the GM coordinate space (N, 3)
 */
void convert_world_to_gm(vector<cv::Mat> xyz,
                         vector<cv::Mat> &xyz_GM)
{

    cv::Point3f trans_lidar2GM = cv::Point3f(0, 0, 1.65111542);
    cv::Vec3f theta_lidar2GM = cv::Vec3f(0, 0, 0);
    // cv::Vec3f theta_lidar2GM = cv::Vec3f(0, 0, M_PI/2);
    cv::Mat T_lidar2GM = buildHomogeneousMatrix(trans_lidar2GM, theta_lidar2GM);

    for (size_t i = 0; i < xyz.size(); i++)
    {
        cv::Mat ones = cv::Mat::ones(xyz[i].rows, 1, CV_32FC1);
        cv::Mat xyz_homo;
        cv::hconcat(xyz[i], ones, xyz_homo);

        // N number of anchors in xyz coord.
        cv::Mat anchors_GM = (T_lidar2GM * xyz_homo.t()).t();     // (4, 4) x (4, N) = (4, N)
        anchors_GM = anchors_GM.colRange(0, anchors_GM.cols - 1); // (N, 3)

        xyz_GM.push_back(anchors_GM); // L x (N, 3)
    }
}
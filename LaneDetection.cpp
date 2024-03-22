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

// extern void detectLanes(VideoCapture inputVideo, VideoWriter outputVideo, int houghStrategy);
extern void detectLanes(sensor_msgs::CompressedImage::ConstPtr msg, HoughTransformHandle* handle, int houghStrategy);
extern void drawLines(Mat &frame, vector<Line> lines);
extern Mat plotAccumulator(int nRows, int nCols, int *accumulator);



std::map<std::string, std::queue<sensor_msgs::CompressedImage::ConstPtr>> imageQueues;
// std::map<std::string, std::queue<sensor_msgs::PointCloud2::ConstPtr>> cloudQueues;

template<typename T>
void callback(const typename T::ConstPtr& msg, const std::string& topicName) {
    // Cast the queue back to the correct type (unsafe, for demonstration only)
    auto& q = imageQueues[topicName];
    q.push(msg);
    ROS_INFO("Message received on %s", topicName.c_str());
}

void imageCallback(const sensor_msgs::CompressedImage::ConstPtr& msg) {
    auto& q = imageQueues["/stereo/left/image_raw/compressed"];
    q.push(msg);
    ROS_INFO("Message received on /stereo/left/image_raw/compressed");
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

    //Setup image callbacks
    // for (const auto& kv : settings["topics"]) {
    //     const YAML::Node& topic_info = kv.second;
    //     std::string topicName = topic_info["name"].as<std::string>();
    //     std::string topicType = topic_info["type"].as<std::string>();
    
    //     // if (topicType == "sensor_msgs/PointCloud2") {
    //     //     imageQueues[topicName] = std::queue<sensor_msgs::PointCloud2::ConstPtr>();
    //     //     ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>(topicName, 10, std::bind(callback<sensor_msgs::PointCloud2>, std::placeholders::_1, topicName));
    //     // } 
    //     if (topicType == "sensor_msgs/CompressedImage") {
    //         imageQueues[topicName] =  std::queue<sensor_msgs::CompressedImage::ConstPtr>();
    //         ros::Subscriber sub = nh.subscribe<sensor_msgs::CompressedImage>(topicName, 10, std::bind(callback<sensor_msgs::CompressedImage>, std::placeholders::_1, topicName));
    //         std::cout << "Subscribed to " << topicName << std::endl;
    //     } // Add more else if blocks for other types
    // }
    imageQueues["/stereo/left/image_raw/compressed"] =  std::queue<sensor_msgs::CompressedImage::ConstPtr>();
    ros::Subscriber sub = nh.subscribe<sensor_msgs::CompressedImage>("/stereo/left/image_raw/compressed", 10, imageCallback);
    ros::Publisher  pub = nh.advertise<sensor_msgs::PointCloud2>("/detected_lanes", 10);

    int houghStrategy = settings["houghStrategy"].as<std::string>() == "cuda" ? CUDA : SEQUENTIAL;
    int frameWidth = settings["frameWidth"].as<int>();
    int frameHeight = settings["frameHeight"].as<int>();

    HoughTransformHandle *handle;
    createHandle(handle, houghStrategy, frameWidth, frameHeight);
    while (ros::ok()) {
        for (auto& kv : imageQueues) {
            std::string topicName = kv.first;
            auto& q = kv.second;
            if (!q.empty()) {
                std::cout << "Processing image from " << kv.first << "\n";
                ROS_INFO("Processing image from %s", topicName.c_str());
                sensor_msgs::CompressedImage::ConstPtr img_msg = q.front();
                q.pop();
                //2 Detect lanes in image
                detectLanes(img_msg, handle, houghStrategy);

                //3 Backprojet 2d lines to 3d using camera calibrations + depth
                

                //4 Convert lane detection to GM format [Ji-Hwan]
                

                //5 Publish detected lanes over ROS [Ji-Hwan]
                sensor_msgs::msg::PointCloud2 pc_msg;

                pub.publish(pc_msg);

            }
        }
        ros::spinOnce();
    }
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

void detectLanes(sensor_msgs::CompressedImage::ConstPtr msg, HoughTransformHandle* handle, int houghStrategy) {
    // Convert to OpenCV image
    cv::Mat frame = cv::imdecode(cv::Mat(msg->data), 1);

    cv::Mat preProcFrame;
    vector<Line> lines;

    preProcFrame = filterLanes(frame);
    preProcFrame = applyGaussianBlur(preProcFrame);
    preProcFrame = applyCannyEdgeDetection(preProcFrame);
    preProcFrame = regionOfInterest(preProcFrame);
    cv::imwrite("roi.png", preProcFrame);

    lines.clear();
    if (houghStrategy == CUDA)
        houghTransformCuda(handle, preProcFrame, lines);
    else if (houghStrategy == SEQUENTIAL)
        houghTransformSeq(handle, preProcFrame, lines);

    drawLines(frame, lines);
    cv::imwrite("frame.png", frame);
}

/**
 * Coordinates the lane detection using the specified hough strategy for the 
 * given input video and writes resulting video to output video
 * 
 * @param inputVideo Video for which lanes are detected
 * @param outputVideo Video where results are written to
 * @param houghStrategy Strategy which should be used to parform hough transform
 */

 /*
void detectLanes(VideoCapture inputVideo, VideoWriter outputVideo, int houghStrategy) {
    Mat frame, preProcFrame;
    vector<Line> lines;

    clock_t readTime = 0;
	clock_t prepTime = 0;
	clock_t houghTime = 0;
	clock_t writeTime = 0;
    clock_t totalTime = -clock();

    int frameWidth = inputVideo.get(CAP_PROP_FRAME_WIDTH);
    int frameHeight = inputVideo.get(CAP_PROP_FRAME_HEIGHT);

    HoughTransformHandle *handle;
    createHandle(handle, houghStrategy, frameWidth, frameHeight);

    cout << "Processing video " << (houghStrategy == CUDA ? "using CUDA" : "Sequentially") << endl;

	for( ; ; ) {
        // Read next frame
        readTime -= clock();
		inputVideo >> frame;
        readTime += clock();
		if(frame.empty())
			break;

        // Apply pre-processing steps
        prepTime -= clock();
        preProcFrame = filterLanes(frame);
        preProcFrame = applyGaussianBlur(preProcFrame);
        preProcFrame = applyCannyEdgeDetection(preProcFrame);
        preProcFrame = regionOfInterest(preProcFrame);
        prepTime += clock();

        // Perform hough transform
        houghTime -= clock();
        lines.clear();
        if (houghStrategy == CUDA)
            houghTransformCuda(handle, preProcFrame, lines);
        else if (houghStrategy == SEQUENTIAL)
            houghTransformSeq(handle, preProcFrame, lines);
        houghTime += clock();

        // Draw lines to frame and write to output video
        writeTime -= clock();
        drawLines(frame, lines);
        outputVideo << frame;
        writeTime += clock();
    }

    destroyHandle(handle, houghStrategy);

    totalTime += clock();
	cout << "Read\tPrep\tHough\tWrite\tTotal (s)" << endl;
	cout << setprecision (4)<<(((float) readTime) / CLOCKS_PER_SEC) << "\t"
         << (((float) prepTime) / CLOCKS_PER_SEC) << "\t"
		 << (((float) houghTime) / CLOCKS_PER_SEC) << "\t"
		 << (((float) writeTime) / CLOCKS_PER_SEC) << "\t"
    	 << (((float) totalTime) / CLOCKS_PER_SEC) << endl;
    cout << "Number of frames " << inputVideo.get(CAP_PROP_FRAME_COUNT) << endl;    
    cout << "Hough average FPS: " << inputVideo.get(CAP_PROP_FRAME_COUNT) / ((double) (houghTime+prepTime) / CLOCKS_PER_SEC) << endl;
    // size_t end_t = clock();
    // cout << "Average in FPS: " << inputVideo.get(CAP_PROP_FRAME_COUNT) / ((double)(end_t - start_t) / CLOCKS_PER_SEC) << endl;
}
*/

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
 * @return uvd Coordinates of the projection point in pixel space (N, 3)
 */
void convert_lines_to_uvd(vector<Line> lines, vector<cv::Point3f> uvd) {

}


/**
 * REFERENCE — https://www.fdxlabs.com/calculate-x-y-z-real-world-coordinates-from-a-single-camera-using-opencv/
 * Project 2D points in pixel space to 3D points in world coordinate space.
 *
 * @param uvd Coordinates of the projection point in pixel space (N, 3)
 * @param T_cam_to_lid Homogeneous transformation matrix from camera to lidar (4, 4)
 * @return xyz - coordinates of a 3D point in the world coordinate space (N, 4)
 */
vector<cv::Point3f> convert_uvd_to_xyz(vector<cv::Point3f> uvd) {
    
    cv::Mat uvd_mat = convert_vec_to_mat(uvd).t();    // (4, N)
    YAML::Node config = YAML::LoadFile("cam.yaml");
    cv::Mat K = config["intrinsic_matrix"].as<cv::Mat>(); // (3, 4) Camera intrinsic matrix
    cv::Mat C = cv::Mat::eye(4, 3, CV_32F);           // (4, 3) Canonical form of matrix

    cv::Mat K_inv = K.inv();
    cv::Mat xyz_mat = K_inv * uvd_mat;

    vector<cv::Point3f> xyz = convert_mat_to_vec(xyz_mat.t()); // (N, 3)

    return xyz
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

    return T
}

/*HELPER FUNCTIONS*/
cv::Mat convert_vec_to_mat(const vector<cv::Point3f>& xyz) {
    cv::Mat mat(xyz.size(), 1, CV_32FC3);
    for (int i = 0; i < xyz.size(); i++) {
        mat.at<cv::Vec3f>(i, 0) = cv::Vec3f(xyz[i].x, xyz[i].y, xyz[i].z, 1);
    }
    return mat; // (N, 4)
}

vector<cv::Point3f> convert_mat_to_vec(const cv::Mat& mat) {
    vector<cv::Point3f> xyz;
    for (int i = 0; i < mat.rows; i++) {
        cv::Vec3f point = mat.at<cv::Vec3f>(i, 0);
        xyz.push_back(cv::Point3f(point[0], point[1], point[2]));
    }
    return xyz; // (N, 3)
}

/**
 * Project 3D points in world coordinate space to 3D points in GM coordinate space.
 *
 * @param xyz Coordinates of a 3D point of lanes in the world coordinate space (N, 4)
 * @return xyz_GM Coordinates of a 3D point of lanes in the GM coordinate space (N, 4)
 */
vector<cv::Point3f> convert_world_to_gm(vector<cv::Point3f> xyz) {

    cv::Mat xyz_mat = convert_vec_to_mat(xyz).t(); // (4, N)
    
    cv::Point3f trans_XX = cv::Point3f(0, 0, 0);
    cv::Vec3f theta_XX = cv::Vec3f(0, 0, 0);

    cv::Point3f trans_YY = cv::Point3f(0, 0, 0);
    cv::Vec3f theta_YY = cv::Vec3f(0, 0, 0);
    
    cv::Point3f trans_ZZ = cv::Point3f(0, 0, 0);
    cv::Vec3f theta_ZZ = cv::Vec3f(0, 0, 0);
    

    cv::Mat T_XX = buildHomogeneousMatrix(trans_XX, theta_XX)
    cv::Mat T_YY = buildHomogeneousMatrix(trans_YY, theta_YY)
    cv::Mat T_ZZ = buildHomogeneousMatrix(trans_ZZ, theta_ZZ)

    cv::Mat xyz_GM_mat = T_ZZ * T_YY * T_XX * xyz_mat;

    vector<cv::Point3f> xyz_GM = convert_mat_to_vec(xyz_GM_mat.t()); // (N, 3)

    return xyz_GM;
}
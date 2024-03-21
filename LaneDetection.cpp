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

extern void detectLanes(VideoCapture inputVideo, VideoWriter outputVideo, int houghStrategy);
extern void detectLanes(sensor_msgs::CompressedImage::ConstPtr msg, HoughTransformHandle* handle, VideoWriter &outputVideo, int houghStrategy);
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

    int houghStrategy = settings["houghStrategy"].as<std::string>() == "cuda" ? CUDA : SEQUENTIAL;
    int frameWidth = settings["frameWidth"].as<int>();
    int frameHeight = settings["frameHeight"].as<int>();

    VideoWriter video(argv[2], VideoWriter::fourcc('a', 'v', 'c', '1') , 10,
                      Size(frameWidth, frameHeight), true);

    HoughTransformHandle *handle;
    createHandle(handle, houghStrategy, frameWidth, frameHeight);
    while (ros::ok()) {
        for (auto& kv : imageQueues) {
            std::string topicName = kv.first;
            auto& q = kv.second;
            if (!q.empty()) {
                std::cout << "Processing image from " << kv.first << "\n";
                ROS_INFO("Processing image from %s", topicName.c_str());
                sensor_msgs::CompressedImage::ConstPtr msg = q.front();
                q.pop();
                //2 Detect lanes in image
                detectLanes(msg, handle, video, houghStrategy);

                //3 Backprojet 2d lines to 3d using camera calibrations + depth

                //4 Convert lane detection to GM format [Ji-Hwan]

                //5 Publish detected lanes over ROS [Ji-Hwan]

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

void detectLanes(
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
}

/**
 * Coordinates the lane detection using the specified hough strategy for the 
 * given input video and writes resulting video to output video
 * 
 * @param inputVideo Video for which lanes are detected
 * @param outputVideo Video where results are written to
 * @param houghStrategy Strategy which should be used to parform hough transform
 */
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

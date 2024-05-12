#include "Line.h"

Line::Line(double theta, double rho, int accum) {
    this->theta = theta;
    this->rho = rho;
    this->accum = accum;
}

/** Calculates y value of line based on given x */
double Line::getY(double x) {
    double thetaRadian = (theta * PI) / 180.0;
        
    return (rho  - x * cos(thetaRadian)) / sin(thetaRadian);
}

/** Calculates x value of line based on given y */
double Line::getX(double y) {
    double thetaRadian = (theta * PI) / 180.0;
        
    return (rho - y * sin(thetaRadian)) / cos(thetaRadian);
}


LineAnchors::LineAnchors(size_t num_anchors) {
    this->anchors = vector<cv::Point3f>(num_anchors, cv::Point3f(0, 0, 0));
}

size_t LineAnchors::getNumAnchors() {
    return this->anchors.size();
}

void LineAnchors::addAnchor(cv::Point3f anchor) {
    this->anchors.push_back(anchor);
}

void LineAnchors::getAnchorsMat(cv::Mat &mat) {
    
    size_t num_anchors = this->getNumAnchors();
   
    for (size_t i = 0; i < num_anchors; i++) {
        mat.at<float>(i, 0) = this->anchors[i].x;
        mat.at<float>(i, 1) = this->anchors[i].y;
        mat.at<float>(i, 2) = this->anchors[i].z;
        mat.at<float>(i, 3) = 1;
    }
}

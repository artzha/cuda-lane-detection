#include "commons.h"

/** Represents Line in frame */
class Line
{

private:
    double theta;
    double rho;
    int confidence; // number of votes in accumulator

public:
    Line(double theta, double rho, int confidence);

    /** Calculates y value of line based on given x */
    double getY(double x);

    /** Calculates x value of line based on given y */
    double getX(double y);

    /** Returns confidence in line */
    int getConfidence() { return this->confidence; }
};

class LineAnchors
{

private:
    vector<cv::Point3f> anchors;

public:
    LineAnchors(size_t num_anchors);

    cv::Point3f operator[](int i) const { return this->anchors[i]; }
    cv::Point3f &operator[](int i) { return this->anchors[i]; }

    /** Get the number of anchors */
    size_t getNumAnchors();

    /** Add anchor */
    void addAnchor(cv::Point3f anchor);

    /** Convert anchors to xyz */
    void getAnchorsMat(cv::Mat &mat);
};
//------------------------------------------------------------------------------
// GRVC MBZIRC Vision
//------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2016 Pablo Ramon Soria - GRVC University of Seville
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//------------------------------------------------------------------------------

// Simple Tag Visualizer
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <vector>

#include <thread>
#include <mutex>

struct ArucoTag {
    unsigned id;
    pcl::PointXYZ position;
    Eigen::Matrix3f rotation;
};


class ZedArucoDetector{
public:	
    bool init(int _cameraIdx, double _tagSize, std::string _calibFile, bool visualize);
    bool stop();

    std::vector<ArucoTag> lastTags();
private:
    Eigen::Matrix3f rotVector2Matrix(const double &_rx, const double &_ry, const double &_rz);

    std::vector<ArucoTag> mLastTags;
    bool mVisualize = false;
	
    std::mutex mSecureMutex;
    std::thread *mProcessThread;
    int mCameraIdx;
    double mTagSize;
    std::string mCalibFile;
    bool mRunning = false;
};

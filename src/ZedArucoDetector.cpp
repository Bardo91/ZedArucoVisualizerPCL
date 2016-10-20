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

#include "ZedArucoDetector.h"

using namespace cv;
using namespace pcl;
using namespace std;

//---------------------------------------------------------------------------------------------------------------------
bool ZedArucoDetector::init(int _cameraIdx, double _tagSize, std::string _calibFile, bool _visualize){

    mCameraIdx = _cameraIdx;
    mTagSize = _tagSize;
    mCalibFile = _calibFile;
    mVisualize = _visualize;

    mProcessThread = new std::thread([&](){
        vector<int>  markerIds;
        vector<vector<Point2f> > markerCorners;
        Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

        if(mVisualize){
            namedWindow("Markers", CV_WINDOW_FREERATIO);
        }

        // Open calibration file
        FileStorage fs(mCalibFile, FileStorage::READ);
        if (!fs.isOpened()) {
            std::cout << "[ERROR] Invalid calibration file or invalid path to calibration file." << std::endl;
        }

        Mat cameraMatrix, distCoeffs;
        fs["MatrixLeft"] >> cameraMatrix;
        fs["DistCoeffsRight"] >> distCoeffs;

        // Init camera
        VideoCapture camera(mCameraIdx);
        if (!camera.isOpened()) {
            std::cout << "[ERROR] Cannot open camera by given index " << mCameraIdx << "!" << std::endl;
            return -1;
        }

        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
        if(mVisualize){
            // Init pcl viewer
           viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("3D Viewer"));
           viewer->setBackgroundColor(0, 0, 0);
           viewer->initCameraParameters();
        }

        mRunning = true;
        while(mRunning){
            // Capture new data
            Mat inputImage;
            camera >> inputImage;
            if (inputImage.rows == 0) {
                break;
            }

            inputImage = inputImage(Rect(0,0,inputImage.cols/2, inputImage.rows));

            // Detect markers
            aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds);
            if(mVisualize){
                aruco::drawDetectedMarkers(inputImage, markerCorners, markerIds);
            }

            // Estimate position markers
            vector<Vec3d> rvecs, tvecs;
            if (markerCorners.size() != 0) {
                aruco::estimatePoseSingleMarkers(markerCorners, mTagSize, cameraMatrix, distCoeffs, rvecs, tvecs);
                if(mVisualize){
                    for (unsigned i = 0; i < rvecs.size(); i++) {
                        cv::aruco::drawAxis(inputImage, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], mTagSize);
                    }
                }
            }

            std::vector<ArucoTag> tags;
            for (unsigned i = 0; i < rvecs.size(); i++) {
                ArucoTag tag;
                tag.id = markerIds[i];
                tag.position = PointXYZ(tvecs[i][0], tvecs[i][1], tvecs[i][2]);
                tag.rotation = rotVector2Matrix(rvecs[i][0], rvecs[i][1], rvecs[i][2]);
                tags.push_back(tag);
            }

            mSecureMutex.lock();
            mLastTags = tags;
            mSecureMutex.unlock();

            if(mVisualize){
                // Show markers on the image
                imshow("Markers", inputImage);
                waitKey(3);

                // Translate to pcl.
                viewer->removeAllCoordinateSystems();
                viewer->addCoordinateSystem(0.1);
                for (unsigned i = 0; i < tags.size(); i++) {
                    ArucoTag tag = tags[i];

                    Eigen::Affine3f pose;
                    pose.matrix().block(0, 3, 3, 1) = Eigen::Vector3f(tvecs[i][0], tvecs[i][1], tvecs[i][2]);
                    pose.matrix().block(0, 0, 3, 3) = tag.rotation;
                    viewer->addCoordinateSystem(0.03, pose, "tag_" + to_string(tag.id));
                }

                viewer->spinOnce(30);
                boost::this_thread::sleep(boost::posix_time::microseconds(30000));
            }
        }
    });
}

//---------------------------------------------------------------------------------------------------------------------
bool ZedArucoDetector::stop(){
    mRunning = false;
    boost::this_thread::sleep(boost::posix_time::seconds(2));
    if(mProcessThread->joinable())
        mProcessThread->join();
}

std::vector<ArucoTag> ZedArucoDetector::lastTags() {
    mSecureMutex.lock();
    auto tags = mLastTags;
    mSecureMutex.unlock();

    return tags;
}

//---------------------------------------------------------------------------------------------------------------------
Eigen::Matrix3f ZedArucoDetector::rotVector2Matrix(const double &_rx, const double &_ry, const double &_rz) {
    Eigen::Vector3f v(_rx, _ry, _rz);
    float norm = v.norm();

    Eigen::Matrix3f rotMat;

    float sv2 = sin(norm / 2);
    float cv2 = cos(norm / 2);

    rotMat(0, 0) = ((_rx*_rx - _ry*_ry - _rz*_rz)*sv2*sv2 + norm*norm*cv2*cv2) / norm / norm;
    rotMat(0, 1) = (2 * sv2*(_rx*_ry*sv2 - norm*_rz*cv2)) / norm / norm;
    rotMat(0, 2) = (2 * sv2*(_rx*_rz*sv2 + norm*_ry*cv2)) / norm / norm;

    rotMat(1, 0) = (2 * sv2*(_rx*_ry*sv2 + norm*_rz*cv2)) / norm / norm;
    rotMat(1, 1) = ((_ry*_ry - _rx*_rx - _rz*_rz)*sv2*sv2 + norm*norm*cv2*cv2) / norm / norm;
    rotMat(1, 2) = (2 * sv2*(_ry*_rz*sv2 - norm*_rx*cv2)) / norm / norm;

    rotMat(2, 0) = (2 * sv2*(_rx*_rz*sv2 - norm*_ry*cv2)) / norm / norm;
    rotMat(2, 1) = (2 * sv2*(_ry*_rz*sv2 + norm*_rx*cv2)) / norm / norm;
    rotMat(2, 2) = ((_rz*_rz - _ry*_ry - _rx*_rx)*sv2*sv2 + norm*norm*cv2*cv2) / norm / norm;


    return rotMat;
}

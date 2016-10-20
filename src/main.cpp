//------------------------------------------------------------------------------
// GRVC MBZIRC Vision
//------------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2016 GRVC University of Seville
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

using namespace std;
using namespace cv;
using namespace pcl;

bool decodeOptions(int _argc, char **_argv, int &_index, double &_size, std::string &_calibFile);
Eigen::Matrix3f rotVector2Matrix(const double &_rx, const double &_ry, const double &_rz);

struct ArucoTag {
	unsigned id;
	PointXYZ position;
	Eigen::Matrix3f rotation;
};


int main(int _argc, char **_argv){
    int index;
    double size;
    std::string calibFile;
    if (!decodeOptions(_argc, _argv, index, size, calibFile)) {
		return -1;
	}

	// Initialize variables
	Mat inputImage; 
	vector<int>  markerIds; 
	vector<vector<Point2f> > markerCorners;
	Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250); 
	namedWindow("Markers", CV_WINDOW_FREERATIO);

    double markerSize = size;
    unsigned cameraIndex = index;

	// Open calibration file
    FileStorage fs(calibFile, FileStorage::READ);
	if (!fs.isOpened()) {
		std::cout << "[ERROR] Invalid calibration file or invalid path to calibration file." << std::endl;
	}

	Mat cameraMatrix, distCoeffs; 
    fs["MatrixLeft"] >> cameraMatrix;
    fs["DistCoeffsRight"] >> distCoeffs;

	// Init camera
	VideoCapture camera(cameraIndex); 
	if (!camera.isOpened()) {
		std::cout << "[ERROR] Cannot open camera by given index " << cameraIndex << "!" << std::endl;
		return -1;
	}

	// Init pcl viewer
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->initCameraParameters();

	for (;;) {
		// Capture new data
		Mat inputImage;
		camera >> inputImage;
		if (inputImage.rows == 0) {
			break;
		}

        inputImage = inputImage(Rect(0,0,inputImage.cols/2, inputImage.rows));

		// Detect markers
		aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds);
		aruco::drawDetectedMarkers(inputImage, markerCorners, markerIds);

		// Estimate position markers
		vector<Vec3d> rvecs, tvecs;
		if (markerCorners.size() != 0) {
			aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);
			for (unsigned i = 0; i < rvecs.size(); i++) {
				cv::aruco::drawAxis(inputImage, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], markerSize);
			}
		}

		// Show markers on the image
		imshow("Markers", inputImage);
		waitKey(3);

		// Translate to pcl.
		std::vector<ArucoTag> tags;
		viewer->removeAllCoordinateSystems();
		viewer->addCoordinateSystem(0.1);
		for (unsigned i = 0; i < rvecs.size(); i++) {
			ArucoTag tag;
			tag.id = markerIds[i];
			tag.position = PointXYZ(tvecs[i][0], tvecs[i][1], tvecs[i][2]);
			tag.rotation = rotVector2Matrix(rvecs[i][0], rvecs[i][1], rvecs[i][2]);
			tags.push_back(tag);

			Eigen::Affine3f pose;
			pose.matrix().block(0, 3, 3, 1) = Eigen::Vector3f(tvecs[i][0], tvecs[i][1], tvecs[i][2]);
			pose.matrix().block(0, 0, 3, 3) = tag.rotation;
			viewer->addCoordinateSystem(0.03, pose, "tag_" + to_string(tag.id));
		}

		viewer->spinOnce(30);
		boost::this_thread::sleep(boost::posix_time::microseconds(30000));
	}
}



bool decodeOptions(int _argc, char**_argv, int &_index, double &_size, string &_calibFile) {
    if(_argc != 4)
        return false;

    _index = atoi(_argv[1]);
    _size = atof(_argv[2]);
    _calibFile = _argv[3];

	return true;
}

//---------------------------------------------------------------------------------------------------------------------
Eigen::Matrix3f rotVector2Matrix(const double &_rx, const double &_ry, const double &_rz) {
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

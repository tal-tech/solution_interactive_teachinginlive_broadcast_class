#include "det_face_mtcnn.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

using namespace facethink;
using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

	if (argc < 6) {
		std::cerr << "Usage: " << argv[0]
			<< " det1_model"
			<< " det2_model"
			<< " det3_model"
			<< " image path"
			<< " config file path" << std::endl;
		return 1;
	}

	const std::string det1_model = argv[1];
	const std::string det2_model = argv[2];
	const std::string det3_model = argv[3];
	const std::string image_path = argv[4];
	const std::string config_file_path = argv[5];

	cv::Mat img = cv::imread(image_path);
	if (img.data == 0) {
		std::cout << "read image failed:" << argv[0] << std::endl;
		// return -1;
	}

	FaceDetectionMTCNN *face_detector = FaceDetectionMTCNN::create(
		det1_model,
		det2_model,
		det3_model,
		config_file_path);
	vector<Rect> rectangles;
	vector<float> confidences;

	face_detector->detection_SCALE(img, rectangles, confidences);
	for (int i = 0; i < rectangles.size(); i++)
	{
		std::cout << "id: " << i << std::endl;
		std::cout << "confidence: " << confidences[i] << std::endl;
		std::cout << "rectangle: " << rectangles[i] << std::endl;
	}

	std::getchar();
	return 0;
}


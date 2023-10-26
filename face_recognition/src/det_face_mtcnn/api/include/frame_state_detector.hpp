#ifndef __FACETHINK_API_FACE_DETECTION_FRAME_STATE_DETECTOR_HPP__
#define __FACETHINK_API_FACE_DETECTION_FRAME_STATE_DETECTOR_HPP__

#define NOMINMAX
#include "config.hpp"
#include <caffe/caffe.hpp>
#include "det_face_mtcnn.hpp"
#include "CLock.h"
#ifdef WIN32
#include "windows.h"
#else
#endif
namespace facethink {
	namespace detfacemtcnn {

		class FrameStateDetector : public FaceDetectionMTCNN {
		public:
			explicit FrameStateDetector(
				const std::string& det1_model_prototxt,
				const std::string& det1_model_binary,
				const std::string& det2_model_prototxt,
				const std::string& det2_model_binary,
				const std::string& det3_model_prototxt,
				const std::string& det3_model_binary,
				const std::string& config_file);

			explicit FrameStateDetector(
				const std::string& det1_model_file,
				const std::string& det2_model_file,
				const std::string& det3_model_file,
				const std::string& config_file);

			virtual void detection_SCALE(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence);
			virtual void detection_SCALE(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence, std::vector<std::vector<cv::Point>>& alignment);

			void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles);
			void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, float scale);
			void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence);
			void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence, std::vector<std::vector<cv::Point>>& alignment);


		protected:

			void Preprocess(const cv::Mat &img);
			void P_Net();
			void R_Net();
			void O_Net();
			void detect_net(int i);

			void nms(std::vector<std::vector<float>> boxes, std::vector<std::vector<float>>& out_boxes, std::vector<int>& pick, std::string type);
			void reGetBoxes();
			void rerec();
			void pad();
			void getRects();

			void cropImages(std::vector<cv::Mat>& cur_imgs, int i);

			void getBoxesByScore(float thresh, int count);
			void bbreg(int count);

			void getPoints();

			void Predict(const cv::Mat& img, int i, int count);
			void Predict(const std::vector<cv::Mat>& imgs, int i);
			void WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels, int i);
			void WrapInputLayer(const std::vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, int i);

			double IoU(const std::vector<float>& rect1, const std::vector<float>& rect2, std::string type);
			void resize_img();
			void GenerateBoxsNew(const cv::Mat& img, int count, std::vector<std::vector<float>>& boxes);

			//param for P, R, O, L net
			//std::vector<std::shared_ptr<Net<float>>> nets_;
			std::shared_ptr<Net<float>> nets_[3];

			std::vector<cv::Size> input_geometry_;
			int num_channels_;

			//variable for the image
			cv::Mat img_;
			std::vector<cv::Mat> img_resized_;

			//variable for the output of the neural network
			std::vector<float> regression_box_temp_;
			std::vector<float> confidence_;
			std::vector<float> confidence_temp_;
			std::vector<std::vector<float>> alignment_;
			std::vector<float> alignment_temp_;

			std::vector<double> scales_;
			std::vector<std::vector<float>> boxes_;
			std::vector<std::vector<float>> params_;

			cv::Mat rect_mat_;
			cv::Mat conf_mat_;

			Config config_;
			//paramter for the threshold
			//int minSize_ = 200;
			//float factor_ = 0.709;
			//float threshold_[3] = { (float)0.5, (float)0.50, (float)0.30};
			//float threshold_NMS_ = 0.5;

			float maximum(float f1, float f2);
			//int maximum(int f1, int f2);
			int minimum(int f1, int f2);
			void sort_vector(std::vector<std::vector<float>>& boxes, std::vector<int>& seq);
			// ������  
			double P_cost_time; 
	  static base::CLock::CCriticalSection m_criticalsection;
    };

	}
}

#endif

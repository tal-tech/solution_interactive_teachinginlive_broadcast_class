#include "frame_state_detector.hpp"
#include <chrono>


#ifdef WIN32
  #include <io.h>
#endif
using namespace base::CLock;

namespace facethink {
	namespace detfacemtcnn {

		CCriticalSection FrameStateDetector::m_criticalsection;
		FrameStateDetector::FrameStateDetector(const std::string& det1_model_prototxt,
			const std::string& det1_model_binary,
			const std::string& det2_model_prototxt,
			const std::string& det2_model_binary,
			const std::string& det3_model_prototxt,
			const std::string& det3_model_binary,
			const std::string& config_file) {

#ifdef WIN32
			if (_access(config_file.c_str(), 0) != -1) {
				config_.ReadIniFile(config_file);
			}
#else
			if (access(config_file.c_str(), 0) != -1) {
				config_.ReadIniFile(config_file);
			}
#endif
			facethinkcaffe::NetBuilder<float> net_builder;
			nets_[0] = net_builder.Create(det1_model_prototxt, det1_model_binary);
			nets_[1] = net_builder.Create(det2_model_prototxt, det2_model_binary);
			nets_[2] = net_builder.Create(det3_model_prototxt, det3_model_binary);

			for (size_t i = 0; i < 3; i++)
			{
				cv::Size input_geometry;
				int num_channel;

				auto input_layer = nets_[i]->blob("data");
				num_channel = input_layer->channels();
				input_geometry = cv::Size(input_layer->width(), input_layer->height());
				input_geometry_.push_back(input_geometry);

				if (i == 0)
					num_channels_ = num_channel;
				else if (num_channels_ != num_channel)
					std::cout << "Error: The number channels of the nets are different!" << std::endl;
			}


		}

		FrameStateDetector::FrameStateDetector(
			const std::string& det1_model_file,
			const std::string& det2_model_file,
			const std::string& det3_model_file,
			const std::string& config_file) {

#ifdef WIN32
			if (_access(config_file.c_str(), 0) != -1) {
				config_.ReadIniFile(config_file);
			}
#else
			if (access(config_file.c_str(), 0) != -1) {
				config_.ReadIniFile(config_file);
			}
#endif
			facethinkcaffe::NetBuilder<float> net_builder;
			nets_[0] = net_builder.Create(det1_model_file);
			nets_[1] = net_builder.Create(det2_model_file);
			nets_[2] = net_builder.Create(det3_model_file);

			for (size_t i = 0; i < 3; i++)
			{
				cv::Size input_geometry;
				int num_channel;

				auto input_layer = nets_[i]->blob("data");
				num_channel = input_layer->channels();
				input_geometry = cv::Size(input_layer->width(), input_layer->height());
				input_geometry_.push_back(input_geometry);

				if (i == 0)
					num_channels_ = num_channel;
				else if (num_channels_ != num_channel)
					std::cout << "Error: The number channels of the nets are different!" << std::endl;
			}
		}

		void FrameStateDetector::detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles)
		{
			Preprocess(img);
			P_Net();
			R_Net();
			O_Net();

			rectangles.clear();
			for (auto &bounding_box : boxes_)
			{
				rectangles.push_back(
					cv::Rect(bounding_box[0] * 1.0 / config_.SCALE,
						bounding_box[1] * 1.0 / config_.SCALE,
						(bounding_box[2] * 1.0 - bounding_box[0] * 1.0) / config_.SCALE,
						(bounding_box[3] * 1.0 - bounding_box[1] * 1.0) / config_.SCALE));
			}
		}

		void FrameStateDetector::detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, float scale)
		{
			Preprocess(img);
			P_Net();
			R_Net();
			O_Net();

			rectangles.clear();
			for (auto &bounding_box : boxes_)
			{
				rectangles.push_back(
					cv::Rect(bounding_box[0] * 1.0 / config_.SCALE,
						bounding_box[1] * 1.0 / config_.SCALE,
						(bounding_box[2] * 1.0 - bounding_box[0] * 1.0) / config_.SCALE,
						(bounding_box[3] * 1.0 - bounding_box[1] * 1.0) / config_.SCALE));
			}
		}

		void FrameStateDetector::detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence)
		{
			detection(img, rectangles);

			confidence = confidence_;
		}

		void FrameStateDetector::detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence, std::vector<std::vector<cv::Point>>& alignment)
		{
			detection(img, rectangles, confidence);

			alignment.clear();
			for (auto &i : alignment_)
			{
				std::vector<cv::Point> temp_alignment;
				for (size_t j = 0; j < i.size() / 2; j++)
				{
					temp_alignment.push_back(cv::Point(i[2 * j] * 1.0 / config_.SCALE, i[2 * j + 1] * 1.0 / config_.SCALE));
				}
				alignment.push_back(std::move(temp_alignment));
			}
		}

		void FrameStateDetector::detection_SCALE(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence, std::vector<std::vector<cv::Point>>& alignment)
		{
			detection_SCALE(img, rectangles, confidence);

			alignment.clear();
			for (auto &i : alignment_)
			{
				std::vector<cv::Point> temp_alignment;
				for (size_t j = 0; j < i.size() / 2; j++)
				{
					temp_alignment.push_back(cv::Point(i[2 * j] * 1.0 / config_.SCALE, i[2 * j + 1] * 1.0 / config_.SCALE));
				}
				alignment.push_back(std::move(temp_alignment));
			}
		}

		void FrameStateDetector::detection_SCALE(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence)
		{
			Lock lock(&m_criticalsection);
			if (img.empty() || img.channels() != 3) {
				BOOST_LOG_TRIVIAL(fatal) << "Input image must has 3 channels.";
			}

			img_resized_.clear();
			scales_.clear();
			params_.clear();
			regression_box_temp_.clear();
			confidence_.clear();
			confidence_temp_.clear();
			alignment_.clear();
			alignment_temp_.clear();
			boxes_.clear();

			BOOST_LOG_TRIVIAL(debug) << "Det Face MTCNN Net: Start Detect.";
			clock_t start = clock();

			//scale image
			cv::Mat scaled_img;
			cv::Size size = cv::Size((int)(img.cols * config_.SCALE), (int)(img.rows * config_.SCALE));
			cv::resize(img, scaled_img, size);
			detection(scaled_img, rectangles, 1.0 / config_.SCALE);
			confidence = confidence_;

			double cost_time = (double(clock() - start)) / CLOCKS_PER_SEC * 1000;
			BOOST_LOG_TRIVIAL(info) << "Det Face MTCNN Net Cost Time: " << cost_time << " ms";

		}

		void FrameStateDetector::Preprocess(const cv::Mat &img)
		{
			/* Convert the input image to the input image format of the network. */
			cv::Mat sample;
			if (img.channels() == 3 && num_channels_ == 1)
				cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
			else if (img.channels() == 4 && num_channels_ == 1)
				cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
			else if (img.channels() == 4 && num_channels_ == 3)
				cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
			else if (img.channels() == 1 && num_channels_ == 3)
				cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
			else
				sample = img;

			cv::Mat sample_float;
			if (num_channels_ == 3)
				sample.convertTo(sample_float, CV_32FC3);
			else
				sample.convertTo(sample_float, CV_32FC1);

			cv::cvtColor(sample_float, sample_float, cv::COLOR_BGR2RGB);
			//sample_float = sample_float;//.t(); //TODO

			img_ = sample_float;
		}

		void FrameStateDetector::P_Net()
		{
			resize_img();
			int count = 0;
			P_cost_time = 0;
			std::vector<std::vector<float>> total_out_boxes;
			std::vector<int> seq; // no use here
			for (auto img_resized : img_resized_) {
				Predict(img_resized, 0, count);
				std::vector<std::vector<float>> boxes;
				GenerateBoxsNew(img_resized, count, boxes);
				std::vector<std::vector<float>> out_boxes;
				//std::cout << "gen boxes size " << boxes.size() << std::endl;
				nms(boxes, out_boxes, seq, "Union");
				count++;
				total_out_boxes.insert(total_out_boxes.end(), out_boxes.begin(), out_boxes.end());
				//std::cout << "total_boxes " << total_out_boxes.size() << std::endl;
			}
            //std::cout << "Forward P-net: " << P_cost_time / 1000 << std::endl;
			nms(total_out_boxes, boxes_, seq, "Union");
			reGetBoxes();
			rerec();
			pad();
		}

		void FrameStateDetector::R_Net()
		{
			detect_net(1);
		}

		void FrameStateDetector::O_Net()
		{
			detect_net(2);
		}

		void FrameStateDetector::cropImages(std::vector<cv::Mat>& cur_imgs, int i)
		{
			for (int j = 0; j < params_.size(); j++) {
				std::vector<float> param = params_[j];
				/*
				*/
				float dy = param[0];
				float edy = param[1];
				float dx = param[2];
				float edx = param[3];
				float y = param[4];
				float ey = param[5];
				float x = param[6];
				float ex = param[7];
				float tmpw = param[8];
				float tmph = param[9];
				//std::cout << "rect " << int(y) << " " << x << " " << int(ey) - int(y) << " " << int(ex) - int(x) << std::endl;
				cv::Rect rect(x, y, ex - x + 1, ey - y + 1);
				cv::Mat src = img_(rect);

				cv::Mat img;
				int top = dy;
				int left = dx;
				if (i == 1)
					img.create(tmph + 1, tmpw + 1, src.type());
				else if (i == 2)
					img.create(tmph, tmpw, src.type());

				img.setTo(cv::Scalar::all(0));
				src.copyTo(img(cv::Rect(left, top, src.cols, src.rows)));

				if (img.size() == cv::Size(0, 0))
					continue;
				if (img.rows == 0 || img.cols == 0)
					continue;
				if (img.size() != input_geometry_[i])
					cv::resize(img, img, input_geometry_[i]);
				img.convertTo(img, CV_32FC3, 0.0078125, -127.5*0.0078125);
				cur_imgs.push_back(img);
			}
		}

		void FrameStateDetector::detect_net(int i)
		{
			float thresh = config_.THRESHOLD[i];  //threshold_[i];
			std::vector<cv::Rect> bounding_box;
			std::vector<float> confidence;
			std::vector<cv::Mat> cur_imgs;
			std::vector<std::vector<cv::Point>> alignment;

			if (params_.size() == 0)
				return;

			cropImages(cur_imgs, i);

			if (cur_imgs.size() == 0)
				return;
			Predict(cur_imgs, i);
			std::vector<int> selected;
			getBoxesByScore(thresh, i);
			if (i == 2)
			{
				getPoints();
			}
			std::vector<int> pick;
			if (i == 1)
			{
				nms(boxes_, boxes_, pick, "Union");
				bbreg(i);
				rerec();
				pad();
			}
			else if (i == 2)
			{
				bbreg(i);
				nms(boxes_, boxes_, pick, "Min");
				// ȥ���Ѿ�nms����points
				std::vector<std::vector<float>> alignment_t;
				std::vector<float> confidence_t;
				for (size_t j = 0; j < pick.size(); j++)
				{
					alignment_t.push_back(alignment_[pick[j]]);
					confidence_t.push_back(confidence_[pick[j]]);
				}
				alignment_ = alignment_t;
				confidence_ = confidence_t;
			}
		}

		void FrameStateDetector::getPoints()
		{
			alignment_.clear();
			for (size_t i = 0; i < boxes_.size(); i++)
			{
				std::vector<float> box = boxes_[i];
				float w = box[3] - box[1] + 1;
				float h = box[2] - box[0] + 1;

				std::vector<float> points;
				for (size_t j = 0; j < 5; j++)
				{
					float x = w * box[j + 9] + box[0] - 1;
					float y = h * box[j + 9 + 5] + box[1] - 1;
					points.push_back(x);
					points.push_back(y);
				}
				alignment_.push_back(points);
			}
		}

		void FrameStateDetector::getBoxesByScore(float thresh, int count)
		{
			std::vector<std::vector<float>> tmp_boxes;
			tmp_boxes.reserve(confidence_temp_.size() / 2);

			for (int j = 0; j < confidence_temp_.size() / 2; j++)
			{
				float score = confidence_temp_[2 * j + 1];
				if (score < thresh) {
					continue;
				}

				std::vector<float> box(19);
				for (size_t k = 0; k < 4; k++)
				{
					box[k] = boxes_[j][k];
				}
				box[4] = score;
				for (size_t i = 0; i < 4; i++)
				{
					box[5 + i] = regression_box_temp_[4 * j + i];
				}
				if (count == 2)
				{
					for (size_t i = 0; i < 10; i++)
					{
						box[9 + i] = alignment_temp_[10 * j + i];
					}
				}

				tmp_boxes.push_back(box);
			}
			boxes_ = tmp_boxes;
		}

		void FrameStateDetector::bbreg(int count)
		{
			for (size_t i = 0; i < boxes_.size(); i++)
			{
				std::vector<float> box = boxes_[i];
				float w = box[2] - box[0] + 1;
				float h = box[3] - box[1] + 1;

				float bb0 = box[0] + box[5] * w;
				float bb1 = box[1] + box[6] * h;
				float bb2 = box[2] + box[7] * w;
				float bb3 = box[3] + box[8] * h;

				boxes_[i][0] = bb0;
				boxes_[i][1] = bb1;
				boxes_[i][2] = bb2;
				boxes_[i][3] = bb3;

				if (count == 2) {
					confidence_.push_back(boxes_[i][4]);
				}
			}
		}

		// ��������
		void  FrameStateDetector::sort_vector(std::vector<std::vector<float>>& boxes, std::vector<int>& seq)
		{
			if (boxes.size() == 0) return;

			// ��¼֮ǰ��˳��
			seq.resize(boxes.size());
			for (size_t i = 0; i < boxes.size(); i++)
			{
				seq[i] = i;
			}

			for (size_t i = 0; i < boxes.size(); i++)
			{
				for (size_t j = i + 1; j < boxes.size(); j++) {
					if (boxes[i][4] < boxes[j][4]) {
						std::vector<float> t = boxes[j];
						boxes[j] = boxes[i];
						boxes[i] = t;

						int tt = seq[j];
						seq[j] = seq[i];
						seq[i] = tt;
					}
				}
			}
		}

		void FrameStateDetector::nms(std::vector<std::vector<float>> boxes, std::vector<std::vector<float>>& out_boxes, std::vector<int>& left, std::string type)
		{
			out_boxes.clear();
			double threshold = config_.THRESHOLD_NMS; //threshold_NMS_;
			std::vector<int> seq;
			sort_vector(boxes, seq);
			while (boxes.size() > 0)
			{
				int j = 0; // ���ֵ
				for (size_t i = 1; ; i++)
				{
					if (i >= boxes.size()) break;

					double a = IoU(boxes[i], boxes[j], type);
					if (a > threshold)
					{
						boxes.erase(boxes.begin() + i);
						seq.erase(seq.begin() + i);
						i--;
					}
				}
				out_boxes.push_back(boxes[j]);
				left.push_back(seq[j]);
				boxes.erase(boxes.begin() + j); //ȥ����ߵ�
				seq.erase(seq.begin() + j);
			}
		}

		void FrameStateDetector::reGetBoxes()
		{
			for (size_t i = 0; i < boxes_.size(); i++)
			{
				std::vector<float> box = boxes_[i];
				float regh = box[3] - box[1];
				float regw = box[2] - box[0];
				float t1 = box[0] + box[5] * regw;
				float t2 = box[1] + box[6] * regh;
				float t3 = box[2] + box[7] * regw;
				float t4 = box[3] + box[8] * regh;
				float t5 = box[4];
				std::vector<float> new_box;
				new_box.push_back(t1);
				new_box.push_back(t2);
				new_box.push_back(t3);
				new_box.push_back(t4);
				new_box.push_back(t5);
				boxes_[i] = new_box;
			}
		}

		void FrameStateDetector::rerec()
		{
			for (size_t i = 0; i < boxes_.size(); i++)
			{
				std::vector<float> box = boxes_[i];
				float w = box[2] - box[0];
				float h = box[3] - box[1];
				float l = w > h ? w : h;
				box[0] = box[0] + w*0.5 - l*0.5;
				box[1] = box[1] + h*0.5 - l*0.5;
				box[2] = box[0] + l;
				box[3] = box[1] + l;
				// C++ fix
				box[0] = box[0] > 0 ? floor(box[0]) : ceil(box[0]);
				box[1] = box[1] > 0 ? floor(box[1]) : ceil(box[1]);
				box[2] = box[2] > 0 ? floor(box[2]) : ceil(box[2]);
				box[3] = box[3] > 0 ? floor(box[3]) : ceil(box[3]);

				boxes_[i] = box;
			}
		}

		float FrameStateDetector::maximum(float f1, float f2)
		{
			return f1 > f2 ? f1 : f2;
		}

		//int FrameStateDetector::maximum(int f1, int f2)
		//{
		//	return f1 > f2 ? f1 : f2;
		//}

		int FrameStateDetector::minimum(int f1, int f2)
		{
			return f1 < f2 ? f1 : f2;
		}

		void FrameStateDetector::pad()
		{
			float w = img_.cols;
			float h = img_.rows;
			params_.clear();
			params_.resize(boxes_.size());
			for (size_t i = 0; i < boxes_.size(); i++)
			{
				std::vector<float> box = boxes_[i];
				float tmph = box[3] - box[1] + 1;
				float tmpw = box[2] - box[0] + 1;
				float dx = 1.;
				float dy = 1.;
				float edx = tmpw;
				float edy = tmph;

				float x = box[0];
				float y = box[1];
				float ex = box[2];
				float ey = box[3];

				if (ex > w)
				{
					edx = -ex + w - 1 + tmpw;
					ex = w - 1;
				}


				if (ey > h) {
					edy = -ey + h - 1 + tmph;
					ey = h - 1;
				}
				if (x < 1) {

					dx = 2 - x;
					x = 1.;
				}
				if (y < 1) {
					dy = 2 - y;
					y = 1.;
				}

				dy = maximum(0, dy - 1);
				dx = maximum(0, dx - 1);
				y = maximum(0, y - 1);
				x = maximum(0, x - 1);
				edy = maximum(0, edy - 1);
				edx = maximum(0, edx - 1);
				ey = maximum(0, ey - 1);
				ex = maximum(0, ex - 1);
				std::vector<float> result;
				result.resize(10);
				result[0] = dy;
				result[1] = edy;
				result[2] = dx;
				result[3] = edx;
				result[4] = y;
				result[5] = ey;
				result[6] = x;
				result[7] = ex;
				result[8] = edx + 1;// tmpw; python�д˴������ô��ݣ���tmpwָ����edx����c++��ֵ����
				result[9] = edy + 1;// tmph;
				params_[i] = result;
			}
		}

		/*
		* Predict function input is a image without crop
		* the reshape of input layer is image's height and width
		*/
		void FrameStateDetector::Predict(const cv::Mat& img, int i, int count1)
		{
			if (img.cols * img.rows == 0)
			{
				BOOST_LOG_TRIVIAL(error) << "Input image is empty.";
				return;
			}

			std::shared_ptr<Net<float>> net = nets_[i];

			auto det_input = net->blob("data");
			std::vector<int> det_input_shape = det_input->shape();
			det_input_shape[0] = 1;
			det_input_shape[1] = num_channels_;
			det_input_shape[2] = img.cols;
			det_input_shape[3] = img.rows;
			det_input->Reshape(det_input_shape);

			std::vector<cv::Mat> input_channels;
			WrapInputLayer(img, &input_channels, i);
			auto time_start = std::chrono::steady_clock::now();
			net->Forward();
			auto time_end = std::chrono::steady_clock::now();
			///BOOST_LOG_TRIVIAL(debug) << "Forward P-net: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() << " ms";
            P_cost_time += (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count());
			auto rect = net->blob("conv4-2");
			auto confidence = net->blob("prob1");
			int count = confidence->count() / 2;

			std::vector<int> det_input_shape1 = confidence->shape();

			const float* rect_begin = rect->cpu_data();
			const float* rect_end = rect_begin + rect->channels() * count;
			regression_box_temp_ = std::vector<float>(rect_begin, rect_end);

			const float* confidence_begin = confidence->cpu_data() + count;
			const float* confidence_end = confidence_begin + count;

			confidence_temp_ = std::vector<float>(confidence_begin, confidence_end);
		}

		/*
		* Predict(const std::vector<cv::Mat> imgs, int i) function
		* used to input is a group of image with crop from original image
		* the reshape of input layer of net is the number, channels, height and width of images.
		*/
		void FrameStateDetector::Predict(const std::vector<cv::Mat>& imgs, int i)
		{
			if (imgs.empty())
			{
				BOOST_LOG_TRIVIAL(error) << "Input images is empty.";
				return;
			}

			std::shared_ptr<Net<float>> net = nets_[i];

			auto det_input = net->blob("data");
			std::vector<int> det_input_shape = det_input->shape();
			det_input_shape[0] = imgs.size();
			det_input_shape[1] = num_channels_;
			det_input_shape[2] = input_geometry_[i].width;
			det_input_shape[3] = input_geometry_[i].height;

			det_input->Reshape(det_input_shape);

			std::vector<cv::Mat> input_channels;
			WrapInputLayer(imgs, &input_channels, i);

			auto time_start = std::chrono::steady_clock::now();
			net->Forward();
			auto time_end = std::chrono::steady_clock::now();
//			if (i == 1)
//				//BOOST_LOG_TRIVIAL(debug) << "Forward R-net: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count()  << " ms";
//				std::cout << "Forward R-net: " << 1.0 * std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1000  << " ms" << std::endl;
//			else if (i == 2)
//				//BOOST_LOG_TRIVIAL(debug) << "Forward O-net: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() << " ms";
//				std::cout << "Forward O-net: " << 1.0 * std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / 1000 << " ms" << std::endl;

			auto confidence = net->blob("prob1");
			int count = confidence->count() / 2; //the channel of confidence is two

			const float* confidence_begin = confidence->cpu_data();
			const float* confidence_end = confidence_begin + count * 2;
			confidence_temp_ = std::vector<float>(confidence_begin, confidence_end);

			//regression_box
			const std::string layers[3] = { "conv4-2","conv5-2","conv6-2" };
			auto rect = net->blob(layers[i]);
			const float* rect_begin = rect->cpu_data();

			const float* rect_end = rect_begin + rect->channels() * count;
			regression_box_temp_ = std::vector<float>(rect_begin, rect_end);

			//landmarks
			if (i == 2) {
				auto points = net->blob("conv6-3");
				const float* points_begin = points->cpu_data();

				const float* points_end = points_begin + points->channels() * count;
				alignment_temp_ = std::vector<float>(points_begin, points_end);
			}
		}

		void FrameStateDetector::WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels, int i)
		{
			auto input_layer = nets_[i]->blob("data");

			int width = input_layer->width();
			int height = input_layer->height();
			float* input_data = input_layer->mutable_cpu_data();
			for (int j = 0; j < input_layer->channels(); ++j)
			{
				cv::Mat channel(height, width, CV_32FC1, input_data);
				input_channels->push_back(channel);
				input_data += width * height;
			}

			cv::split(img.t(), *input_channels);

		}

		/*
		* WrapInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, int i) function
		* used to write the separate BGR planes directly to the input layer of the network
		*/
		void FrameStateDetector::WrapInputLayer(const std::vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, int i)
		{
			//Blob<float> *input_layer = nets_[i]->input_blobs()[0];
			auto input_layer = nets_[i]->blob("data");

			int width = input_layer->width();
			int height = input_layer->height();
			int num = input_layer->num();
			float *input_data = input_layer->mutable_cpu_data();

			for (int j = 0; j < num; j++) {
				for (int k = 0; k < input_layer->channels(); ++k) {
					cv::Mat channel(height, width, CV_32FC1, input_data);
					input_channels->push_back(channel);
					input_data += width * height;
				}
				cv::Mat img = imgs[j].t();
				cv::split(img, *input_channels);
				input_channels->clear();
			}
		}

		double FrameStateDetector::IoU(const std::vector<float>& rect1, const std::vector<float>& rect2, std::string type)
		{
			int xx1 = maximum(rect1[0], rect2[0]);
			int yy1 = maximum(rect1[1], rect2[1]);
			int xx2 = minimum(rect1[2], rect2[2]);
			int yy2 = minimum(rect1[3], rect2[3]);
			int w = maximum(0, xx2 - xx1 + 1);
			int h = maximum(0, yy2 - yy1 + 1);
			int inter = w * h;
			int area1 = (rect1[2] - rect1[0] + 1) * (rect1[3] - rect1[1] + 1);
			int area2 = (rect2[2] - rect2[0] + 1) * (rect2[3] - rect2[1] + 1);
			if (type == "Min") {
				int l = area1 > area2 ? area1 : area2;
				int s = area1 < area2 ? area1 : area2;
				return 1.0 * inter / s;
			}
			else {
				double o = (double)inter / ((double)area1 + (double)area2 - (double)inter);
				return o;
			}
		}

		void FrameStateDetector::resize_img()
		{
			cv::Mat img = img_;
			long height = (long)img.rows;
			long width = (long)img.cols;

			int minSize = config_.MINSIZE;
			double factor = config_.FACTOR;

			double scale = (double)12. / minSize;
			int minWH = std::min(height, width) * scale;

			std::vector<cv::Mat> img_resized;

			int count = 0;
			while (minWH >= 12)
			{
				double p = pow(factor, count);
				double fh = height * scale * p;
				double fw = width * scale * p;
				int resized_h = (int)ceil(fh);
				int resized_w = (int)ceil(fw);
				//printf("p is %0.17f\n", scale * p);
				//printf("h is %0.17f\n", fh);
				/*if (count == 0) {
				resized_h = 48; resized_w = 64;
				}else if (count == 1) {
				resized_h = 37; resized_w =49;
				}
				else if (count == 2) {
				resized_h = 27; resized_w = 36;
				}
				else if (count == 3) {
				resized_h = 21; resized_w = 27;
				}
				else if (count == 4) {
				resized_h = 16; resized_w = 21;
				}*/
				cv::Mat resized;
				cv::resize(img, resized, cv::Size(resized_w, resized_h));
				resized.convertTo(resized, CV_32FC3, 0.0078125, -127.5*0.0078125);
				img_resized.push_back(resized);

				scales_.push_back(scale * pow(factor, count));
				minWH *= factor;
				count++;
			}

			img_resized_ = img_resized;
		}

		void FrameStateDetector::GenerateBoxsNew(const cv::Mat& img, int count, std::vector<std::vector<float>>& boxes)
		{
			boxes.clear();
			int image_h = img.rows;
			int image_w = img.cols;
			double thresh = config_.THRESHOLD[0];  //threshold_[0];
			int stride = 2;
			int cellSize = input_geometry_[0].width;
			double scale = scales_[count];

			int feature_map_h = std::ceil((double)(image_h - cellSize)*1.0 / stride) + 1;
			int feature_map_w = std::ceil((double)(image_w - cellSize)*1.0 / stride) + 1;

			int layer_count = feature_map_h * feature_map_w;
			for (size_t i = 0; i < feature_map_h; i++)
			{
				double socre = 0;
				for (size_t j = 0; j < feature_map_w; j++) {
					socre = confidence_temp_[j * feature_map_h + i];
					if (socre < thresh)
						continue;

					double ddx1 = (double)(stride * j + 1) / (double)scale;
					double ddy1 = (double)(stride * i + 1) / (double)scale;
					double ddx2 = (double)(stride * j + cellSize) / (double)scale;
					double ddy2 = (double)(stride * i + cellSize) / (double)scale;

					float x1 = std::floor(ddx1);
					float y1 = std::floor(ddy1);
					float x2 = std::floor(ddx2);
					float y2 = std::floor(ddy2);

					float dx1 = regression_box_temp_[j * feature_map_h + i + layer_count * 0];
					float dy1 = regression_box_temp_[j * feature_map_h + i + layer_count * 1];
					float dx2 = regression_box_temp_[j * feature_map_h + i + layer_count * 2];
					float dy2 = regression_box_temp_[j * feature_map_h + i + layer_count * 3];

					std::vector<float> one_box;
					one_box.reserve(9);
					one_box.push_back(x1);
					one_box.push_back(y1);
					one_box.push_back(x2);
					one_box.push_back(y2);
					one_box.push_back(socre);
					one_box.push_back(dx1);
					one_box.push_back(dy1);
					one_box.push_back(dx2);
					one_box.push_back(dy2);
					boxes.push_back(one_box);
				}
			}
		}
	}
}

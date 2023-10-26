
#include "det_face_mtcnn.hpp"
#include "det_face_id.hpp"
#include<opencv2/imgproc/imgproc_c.h>

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <vector>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/filesystem.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/progress.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/lexical_cast.hpp>


using namespace facethink;

//设置log格式
void setupLog(std::string filename) {
    typedef boost::log::sinks::synchronous_sink< boost::log::sinks::text_file_backend > sink_t;
    boost::property_tree::ptree pt;
    boost::shared_ptr< sink_t > file_sink = boost::log::add_file_log
            (
                    boost::log::keywords::auto_flush = true,
                    boost::log::keywords::file_name = filename,
                    boost::log::keywords::format = "[%TimeStamp%]: %Message%"
            );
    boost::log::add_common_attributes();
    int log_level = pt.get<int>("log_level", 0);
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= log_level);
}

//遍历目录内文件或文件夹，非循环迭代
std::vector<std::string>getFilePath(std::string folder_path) {
    namespace fs = boost::filesystem;
    fs::directory_iterator end;
    int file_count = 0;
    std::vector<std::string>filePaths;
    for (fs::directory_iterator dir(folder_path); dir != end; dir++)
    {
        std::string fn = dir->path().string();
        //std::cout << fn << std::endl;
        filePaths.push_back(fn);
    }
    return filePaths;
}

//分割字符,获取图片名作为注册ID
int segmentString(std::string str) {
    std::vector<std::string> dst;
    boost::split(dst, str, boost::is_any_of("/."), boost::token_compress_on);
    //for (size_t i = 0; i< dst.size(); i++){std::cout << dst[i] << std::endl;}
    if (dst.size() == 0) { return -1; }
    while (dst.back().compare("jpg") == 0) //jpg
    {
        dst.pop_back();
    }
    std::string img_name = dst.back();
    int idx = boost::lexical_cast<int>(img_name);
    return idx;
}


int main(int argc, char *argv[]){

    setupLog("./log/det_face_id_test_ges_nomeg_1.log"); //设置日志格式

    // 参数
	const std::string det_model = "/home/guoweiye/workspace/face_recognition/src/det_face_id/model/";
	const std::string config_file_path = "/home/guoweiye/workspace/face_recognition/src/det_face_id/model/config.ini";
    const std::string gallery_imgs_folder = "/home/guoweiye/workspace/face_recognition/src/det_face_id/images/gallery/";
    const std::string test_imgs_folder = "/home/guoweiye/workspace/face_recognition/src/det_face_id/images/testing/";
    const int max_loop = 1;

    FaceDetectionMTCNN *face_detector = FaceDetectionMTCNN::create(
            det_model + "det_face_mtcnn_1_v2.0.0.bin",
            det_model + "det_face_mtcnn_2_v2.0.0.bin",
            det_model + "det_face_mtcnn_3_v2.0.0.bin",
            config_file_path);

	FaceIDDetection *faceID_detector = FaceIDDetection::create(
            det_model + "det_face_id_v1.0.3-1.trt",
			config_file_path);
	int loop = 0;
    while(max_loop == -1 || loop < max_loop) {
        // Register process
        std::vector<std::string>gallery_imgs_path = getFilePath(gallery_imgs_folder);
        std::map<int, std::vector<cv::Mat>> id_features;
        for (int i = 0; i < gallery_imgs_path.size(); i++)
        {
            std::string img_path = gallery_imgs_path[i];

            std::vector<std::string>test_imgs_path = getFilePath(img_path);
            for (int j = 0; j < test_imgs_path.size(); j++) {
                std::cout << "regist image " << test_imgs_path[j] << std::endl;
                cv::Mat img = cv::imread(test_imgs_path[j]);
                //cv::Mat img = cv::imread("/home/tal/work/det_face_id/data/images/2084.jpg");
                if (img.data == 0) {
                    BOOST_LOG_TRIVIAL(error) << "read image failed:" << img_path;
                    continue;
                }
                int idx = segmentString(img_path);
                std::vector<cv::Rect> rectangles;
                std::vector<float> confidences;
                std::vector<std::vector<cv::Point>> alignment;
                face_detector->detection_SCALE(img, rectangles, confidences, alignment);

                std::vector<cv::Mat> features;
                int ret = faceID_detector->calc_feature(img, rectangles, alignment, features);
                if (ret < 0) {
                    BOOST_LOG_TRIVIAL(error) << "Regist Image error:" << test_imgs_path[j] << " Error Code is:" << ret;
                } else {
                    id_features[idx].push_back(features[0]);
                }
            }
        }
        std::cout << "Regist over" << std::endl;
        // Feature merge process
        std::vector<cv::Mat> target_features;
        std::vector<size_t> target_ids;

        for (auto item = id_features.begin(); item != id_features.end(); item++){
            int id = item->first;
            auto features = item->second;
            cv::Mat merged_feature;
            // faceID_detector->merge_feature(features, merged_feature);
            target_ids.push_back(id);
            // target_features.push_back(merged_feature);
            target_features.push_back(features[0]);
        }

        std::cout << "merge_feature over" << std::endl;
        // Detection process
        std::vector<std::string>test_folders = getFilePath(test_imgs_folder);
        for (int i = 0; i < test_folders.size(); i++)
        {
            std::string img_path = test_folders[i];
            std::cout << "det image " << img_path << std::endl;
            cv::Mat img = cv::imread(img_path);
            std::vector<cv::Rect> rectangles;
            std::vector<float> confidences;
            std::vector<std::vector<cv::Point>> alignment;
            face_detector->detection_SCALE(img, rectangles, confidences, alignment);

            std::vector<cv::Mat> features;
            int ret = faceID_detector->calc_feature(img, rectangles, alignment, features);

            std::vector<int> detect_ids;
            std::vector<float> detect_conf;
            faceID_detector->get_ids(features, target_ids, target_features, detect_ids, detect_conf);

            for (int j = 0; j < rectangles.size(); j++) {
                cv::rectangle(img, rectangles[j], {0, 0, 255}, 1);
                std::string label = std::to_string(detect_ids[j]) + " " + std::to_string(detect_conf[j]);
                cv::putText(img, label, {rectangles[j].x, rectangles[j].y}, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, {0, 255, 0}, 1, CV_AA);
                BOOST_LOG_TRIVIAL(info) << "Det Face:" << img_path << " " <<
                                                                   rectangles[j].x << " " <<
                                                                   rectangles[j].y << " " <<
                                                                   rectangles[j].width << " " <<
                                                                   rectangles[j].height << " " <<
                                                                   detect_ids[j] << " " <<
                                                                   detect_conf[j];
            }
            cv::imwrite("./rslt/rs_nomeg_1/" + std::to_string(i) + ".jpg", img);
        }
        std::cout << "Current loop" << loop++ << std::endl;
    }
    return 0;
}

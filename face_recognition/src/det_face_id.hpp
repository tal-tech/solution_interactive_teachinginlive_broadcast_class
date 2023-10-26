///////////////////////////////////////////////////////////////////////////////////////
///  Copyright (C) 2017, TAL AILab Corporation, all rights reserved.
///
///  @file: det_face_id.hpp
///  @brief 检测图像中的人脸身份ID
///  @details 最初版本
//
//
///  @version 2.0.0.8
///  @author Ye Gensheng
///  @date 2019-01-21
///
///  @see 使用参考：demo.cpp
///
///////////////////////////////////////////////////////////////////////////////////////

#ifndef __FACETHINK_API_DET_FACE_ID_HPP__
#define __FACETHINK_API_DET_FACE_ID_HPP__

#include <string>
#include <math.h>
#include <opencv2/opencv.hpp>

#ifdef WIN32
#ifdef DLL_EXPORTS
#define EXPORT_CLASS   __declspec(dllexport)
#define EXPORT_API  extern "C" __declspec(dllexport)
#define EXPORT_CLASS_API

#else
#define EXPORT_CLASS   __declspec(dllimport )
#define EXPORT_API  extern "C" __declspec(dllimport )
#endif
#else
#define EXPORT_CLASS
#define EXPORT_API  extern "C" __attribute__((visibility("default")))
#define EXPORT_CLASS_API __attribute__((visibility("default")))
#endif

namespace facethink {

	class EXPORT_CLASS FaceIDDetection {
	public:

		EXPORT_CLASS_API explicit FaceIDDetection(void);
		EXPORT_CLASS_API virtual ~FaceIDDetection(void);

		/*!
		\brief SDK初始化函数，必须先于任何其他SDK函数之前调用，create的重载函数。
		@param [in] det_model_file 指定SDK对应的模型文件路径。
		@param [in] config_file 指定SDK对应的参数配置文件路径,详情见config.ini文件。
		@return
		@remarks 初始化函数需要读取模型等文件，需要一定时间等待。
		*/
		EXPORT_CLASS_API static FaceIDDetection* create(
			const std::string& det_model_file,
			const std::string& config_file);

		/// \brief 人脸注册，注册的顺序记为人脸ID，如：第一个注册的人脸ID为0，第二个注册的人脸ID为1，依次类推。
		/// @param [in] img 输入的图像数据，只支持如下一种种格式:
		/// - 1.BGR图：img为一维数组，每个元素（字节）表示一个像素点的单通道取值，三个连续元素表示一个像素点的三通道取值，顺序为BGR。
		/// @param [in] id，默认值为-1时，表示当前为新人脸注册，当设定id（大于等于0）时，表示更新之前的注册人脸图像。
		/// @param [in] is_rgb_format，默认值为false时，表示图片格式为BGR。
		/// @return
		/// @retval >= 0 注册成功,返回注册的人脸ID值。
		/// @retval -1 提取人脸图像特征失败。
		/// @retval -2 更新注册人脸图像失败。
		/// @retval -3 注册集达到最大容量。
		/// @retval -4 输入图像错误
		EXPORT_CLASS_API  virtual int registration(const cv::Mat& img, const int id = -1, bool is_rgb_format = false) = 0;

		/// \brief 识别人脸ID，检测的id为注册的顺序。
		/// @param [in] img 输入的图像数据，只支持如下一种种格式:
		/// - 1.BGR图：img为一维数组，每个元素（字节）表示一个像素点的单通道取值，三个连续元素表示一个像素点的三通道取值，顺序为BGR。
		/// @param [out] id，识别的人脸ID，当id = -1时，表示识别的人脸没有在注册集中注册。
		/// @param [in] is_rgb_format，默认值为false时，表示图片格式为BGR。
		/// @return
		/// @retval 0 人脸识别成功。
		/// @retval -1 提取人脸图像特征失败。
		/// @retval -4 输入图像错误。
		EXPORT_CLASS_API virtual int detection(const cv::Mat& img, int& id, bool is_rgb_format = false) = 0;

		/// \brief 输出模型处理后的特征。
		/// @param [in] img 输入的图像数据，只支持如下一种种格式:
		/// - 1.BGR图：img为一维数组，每个元素（字节）表示一个像素点的单通道取值，三个连续元素表示一个像素点的三通道取值，顺序为BGR。
		/// @param [out] id，识别的人脸ID，当id = -1时，表示识别的人脸没有在注册集中注册。
		/// @param [in] is_rgb_format，默认值为false时，表示图片格式为BGR。
		/// @return
		/// @retval 0 人脸识别成功。
		/// @retval -1 提取人脸图像特征失败。
		/// @retval -4 输入图像错误。
		EXPORT_CLASS_API virtual int detection_Feature(const cv::Mat& img, cv::Mat& out_mat, bool is_rgb_format = false) = 0;

		/// \brief 输出模型处理后的特征。
		/// @param [in] img 输入的图像数据，只支持如下一种种格式:
		/// - 1.BGR图：img为一维数组，每个元素（字节）表示一个像素点的单通道取值，三个连续元素表示一个像素点的三通道取值，顺序为BGR。
		/// @param [out] id，识别的人脸ID，当id = -1时，表示识别的人脸没有在注册集中注册。
		/// @param [out] out_mat, 模型处理后的人脸特征
		/// @param [in] is_rgb_format，默认值为false时，表示图片格式为BGR。
		/// @return
		/// @retval 0 人脸识别成功。
		/// @retval -1 提取人脸图像特征失败。
		/// @retval -4 输入图像错误。
		EXPORT_CLASS_API virtual int detection_withFeature(const cv::Mat& img, int& id, cv::Mat& out_mat, bool is_rgb_format = false) = 0;

		/// \brief 输出模型处理后的特征。
		/// @param [in] img 输入的图像数据，只支持如下一种种格式:
		/// - 1.BGR图：img为一维数组，每个元素（字节）表示一个像素点的单通道取值，三个连续元素表示一个像素点的三通道取值，顺序为BGR。
		/// @param [out] id，识别的人脸ID，当id = -1时，表示识别的人脸没有在注册集中注册。
		/// @param [out] out_mat, 模型处理后的人脸特征
		/// @param [out] gallery_id 注册的人脸ID，顺序对应dist
		/// @param [out] dist 输入人脸与注册人脸距离，顺序与gallery_id对应。
		/// @param [in] is_rgb_format，默认值为false时，表示图片格式为BGR。
		/// @return
		/// @retval 0 人脸识别成功。
		/// @retval -1 提取人脸图像特征失败。
		/// @retval -4 输入图像错误。
		EXPORT_CLASS_API virtual int detection_withFeature(const cv::Mat& img, int& id, cv::Mat& out_mat, std::vector<int>& gallery_id, std::vector<double>& dist, bool is_rgb_format = false) = 0;

		/// \brief 删除人脸注册ID
		/// @param[in] id 为注册返回的id
		///@retval = 0 删除成功
		///@retval = -1 无效id
		EXPORT_CLASS_API virtual int delFaceid(int id) = 0;

		/// \brief 清空已注册数据
		EXPORT_CLASS_API virtual void clear_registration() = 0;

		/// \brief 测试接口
		/// @param [in] img 输入的图像数据
		/// @param [out] id，识别的人脸ID，当id = -1时，表示识别的人脸没有在注册集中注册
		/// @param [out] score, 特征最近距离
		/// @retval 0 人脸识别成功。
		/// @retval -1 提取人脸图像特征失败。
		/// @retval -4 输入图像错误。
		EXPORT_CLASS_API virtual int detection_test(const cv::Mat& img, int& id, double& score, bool is_rgb_format = false) = 0;

        /// \brief 检测输入图片的特征 For GES
        /// @param [in] img 输入的图像数据
        /// @param [in] rects mtcnn算法检测出来的人脸框
        /// @param [in] points mtcnn算法检测出来的人脸关键点
        /// @param [out] features，识别的人脸特征
        /// @param [in] is_rgb_format, 默认值为false时，表示图片格式为BGR。
        /// @retval 0 特征检测成功。
        /// @retval -1 输入人脸框数量为0。
		EXPORT_CLASS_API virtual int calc_feature(
		        const cv::Mat & img,
		        const std::vector<cv::Rect> & rects,
		        const std::vector<std::vector<cv::Point>> & points,
		        std::vector<cv::Mat> & features,
		        bool is_rgb_format = false) = 0;

        /// \brief 检测输入图片的特征 For GES
        /// @param [in] img 输入的图像数据
        /// @param [in] rects mtcnn算法检测出来的人脸框
        /// @param [in] points mtcnn算法检测出来的人脸关键点
        /// @param [out] features，识别的人脸特征
        /// @param [out] crop_images，对齐后的人脸图片
        /// @param [in] is_rgb_format, 默认值为false时，表示图片格式为BGR。
        /// @retval 0 特征检测成功。
        /// @retval -1 输入人脸框数量为0。
        EXPORT_CLASS_API virtual int calc_feature(
                const cv::Mat & img,
                const std::vector<cv::Rect> & rects,
                const std::vector<std::vector<cv::Point>> & points,
                std::vector<cv::Mat> & features,
                std::vector<cv::Mat> & crop_images,
                bool is_rgb_format = false) = 0;

        /// \brief 特征融合 For GES
        /// @param [in] features 待融合的图像特征
        /// @param [out] merged_feature，融合之后的人脸特征向量
        /// @retval 0 执行成功。
        EXPORT_CLASS_API virtual int merge_feature(const std::vector<cv::Mat> & features, cv::Mat& merged_feature) = 0;

        /// \brief 检测输入图片的特征 For GES
        /// @param [in] unknown_features 检测出来，待匹配的人人脸特征
        /// @param [in] target_ids 已知的人脸ID
        /// @param [in] target_features 已知的人脸特征，与target_ids顺序要一一对应
        /// @param [out] detect_ids，识别的人脸ID，-1表示未识别到
        /// @param [out] detect_conf, 识别的人脸置信度，与detect_ids一一对应。
        /// @retval 0 特征检测成功。
        /// @retval -1 输入人脸框数量为0。
        EXPORT_CLASS_API virtual int get_ids(const std::vector<cv::Mat> & unknown_features, const std::vector<size_t> & target_ids, const std::vector<cv::Mat> & target_features, std::vector<int> & detect_ids, std::vector<float> & detect_conf) = 0;


	};
}

#endif

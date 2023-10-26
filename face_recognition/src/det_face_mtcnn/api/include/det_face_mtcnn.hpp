///////////////////////////////////////////////////////////////////////////////////////
///  Copyright (C) 2017, TAL AILab Corporation, all rights reserved.
///
///  @file: det_face_mtcnn.hpp
///  @brief 检测图像中的人脸
///  @details
///
///
///  @version 2.0.0.2
///  @author Ye Gensheng
///  @date 2018-04-20
///  
///  @see 使用参考：demo.cpp
///
///////////////////////////////////////////////////////////////////////////////////////

#ifndef __FACETHINK_API_DET_FACE_MTCNN_HPP__
#define __FACETHINK_API_DET_FACE_MTCNN_HPP__

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

    class EXPORT_CLASS FaceDetectionMTCNN {
    public:

		EXPORT_CLASS_API explicit FaceDetectionMTCNN(void);
		EXPORT_CLASS_API virtual ~FaceDetectionMTCNN(void);

		/*!
		\brief SDK初始化函数，必须先于任何其他SDK函数之前调用。
		@param [in] det1_model_prototxt 指定SDK对应的模型1网络文件路径。
		@param [in] det1_model_binary 指定SDK对应的模型1参数文件路径。
		@param [in] det2_model_prototxt 指定SDK对应的模型2网络文件路径。
		@param [in] det2_model_binary 指定SDK对应的模型2参数文件路径。
		@param [in] det3_model_prototxt 指定SDK对应的模型3网络文件路径。
		@param [in] det3_model_binary 指定SDK对应的模型3参数文件路径。
		@param [in] config_file 指定SDK对应的参数配置文件路径。
		@return
		@remarks 初始化函数需要读取模型等文件，需要一定时间等待。
		*/
		EXPORT_CLASS_API static FaceDetectionMTCNN* create(
			const std::string& det1_model_prototxt,
			const std::string& det1_model_binary,
			const std::string& det2_model_prototxt,
			const std::string& det2_model_binary,
			const std::string& det3_model_prototxt,
			const std::string& det3_model_binary,
			const std::string& config_file);


		/*!
		\brief SDK初始化函数，必须先于任何其他SDK函数之前调用，create的重载函数。
		@param [in] det1_model_file 指定SDK对应的模型文件1路径。
		@param [in] det2_model_file 指定SDK对应的模型文件2路径。
		@param [in] det3_model_file 指定SDK对应的模型文件3路径。
		@param [in] config_file 指定SDK对应的参数配置文件路径,详情见config.ini文件。
		@return
		@remarks 初始化函数需要读取模型等文件，需要一定时间等待。
		*/
		EXPORT_CLASS_API static FaceDetectionMTCNN* create(
			const std::string& det1_model_file,
			const std::string& det2_model_file,
			const std::string& det3_model_file,
			const std::string& config_file);


		/// \brief 检测输入图像中的人脸（检测时图像大小默认压缩50%，可通过配置文件修改参数）。
		/// @param [in] img 输入的图像数据，支持如下两种种格式:
		/// - 1.灰度图：img为一维数组，每个元素（字节）表示一个像素点的灰度值(范围为0-255）。
		/// - 2.BGR图：img为一维数组，每个元素（字节）表示一个像素点的单通道取值，三个连续元素表示一个像素点的三通道取值，顺序为BGR。
		/// @param [out] rectangles 保存SDK检测到的人脸矩形框，容器中每个元素表示一个人的检测结果。
		/// @param [out] confidence 保存SDK检测到的人脸框置信度，其中每个元素与rectangles中的元素对应。
		/// @return
		EXPORT_CLASS_API virtual void detection_SCALE(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence) = 0;


		/// \brief 检测输入图像中的人脸（检测时图像大小默认压缩50%，可通过配置文件修改参数）。
		/// @param [in] img 输入的图像数据，支持如下两种种格式:
		/// - 1.灰度图：img为一维数组，每个元素（字节）表示一个像素点的灰度值(范围为0-255）。
		/// - 2.BGR图：img为一维数组，每个元素（字节）表示一个像素点的单通道取值，三个连续元素表示一个像素点的三通道取值，顺序为BGR。
		/// @param [out] rectangles 保存SDK检测到的人脸矩形框，容器中每个元素表示一个人的检测结果。
		/// @param [out] confidence 保存SDK检测到的人脸框置信度，其中每个元素与rectangles中的元素对应。
		/// @param [out] alignment 保存SDK检测到的人脸上5个点（两眼球+两嘴角+鼻尖）。
		/// @return
		EXPORT_CLASS_API virtual void detection_SCALE(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence, std::vector<std::vector<cv::Point>>& alignment) = 0;
		
    };

}

#endif

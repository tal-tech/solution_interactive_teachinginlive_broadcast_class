#include "FaceBodyPlugin.h"
#include "base64.h"
#include <chrono>   
#include <iomanip>
#include "json/document.h"  
#include "json/prettywriter.h"
using namespace rapidjson;
using namespace std;
using namespace chrono;
#ifdef WIN32
#pragma comment( lib,"winmm.lib" )
#else
typedef unsigned long       DWORD;
#include <time.h> 
#endif
using namespace base::base64;


const int BUFFER_SIZE = 1024 * 1024 * 10;
FaceBodyPlugin::FaceBodyPlugin()
{
	m_pBuffer1 = new char[BUFFER_SIZE];
	m_pBuffer2 = new char[BUFFER_SIZE];
}


FaceBodyPlugin::~FaceBodyPlugin()
{
}



int FaceBodyPlugin::InvokeMethod(const char* strCommand,
	const std::map<std::string, std::string>& args,
	std::map<std::string, std::string>& out
)
{
	
	return -1;
}


int FaceBodyPlugin::InvokeMethod(const char* strCommand,
	const char* args,
	char* out
)
{
	try {
		if (strcmp("face.detect.multi",strCommand) == 0)
		{
			Document document;
			document.Parse<0>(args);
			if (document.HasParseError())
			{
				std::cout << "Json Parse error: " << std::endl;
				return 1;
			}

			if (!document.HasMember("img"))
			{
				std::cout << "Json Parse error: " << std::endl;
				return 1;
			}

			Value &img = document["img"];

			if (!img.IsString())
			{
				std::cout << "Json Parse error: " << std::endl;
				return 1;
			}
			std::string imgbase64 = img.GetString();
			int len = calc_data_len(imgbase64.c_str(), imgbase64.length());
			//char * pOut = new char[len + 1];
			memset(m_pBuffer1, 0, BUFFER_SIZE);
			

			//auto start = system_clock::now();
			base64_decode(imgbase64.c_str(), imgbase64.length(), m_pBuffer1);
			//auto end = system_clock::now();
			//std::cout << "cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
			std::vector<char> data(m_pBuffer1, m_pBuffer1 + len);
			//delete[] pOut;

			cv::Mat image = cv::imdecode(data, CV_LOAD_IMAGE_COLOR);

			std::vector<cv::Rect> rectangles;
			std::vector<float> confidence;

			m_body_detector_server->detection_SCALE(image, rectangles, confidence);

			Document jsondoc;
			jsondoc.SetObject();
			Document::AllocatorType &allocator = jsondoc.GetAllocator();

			Value rectanglesArray(rapidjson::kArrayType);
			Value confidenceArray(rapidjson::kArrayType);

			for (int i = 0; i < rectangles.size(); i++)
			{
				std::stringstream stream;

				int x = rectangles[i].x;
				int y = rectangles[i].y;
				int width = rectangles[i].width;
				int height = rectangles[i].height;
				rapidjson::Value object(rapidjson::kObjectType);
				object.AddMember("x", x, allocator);
				object.AddMember("y", y, allocator);
				object.AddMember("width", width, allocator);
				object.AddMember("height", height, allocator);
				rectanglesArray.PushBack(object, allocator);

				std::stringstream stream2;
				stream2 << confidence[i];
				std::string string_temp2 = stream2.str();   //此处也可以用 stream>>string_temp  

				confidenceArray.PushBack(rapidjson::Value((char*)string_temp2.c_str(), allocator), allocator);
			}
			jsondoc.AddMember("rectangles", rectanglesArray, allocator);    //添加数组
			jsondoc.AddMember("confidence", confidenceArray, allocator);    //添加数组
			StringBuffer buffer;
			Writer<StringBuffer> writer(buffer);
			jsondoc.Accept(writer);

			strcpy(out,buffer.GetString());
		}
		else if (strcmp("face.detect.multi2",strCommand) == 0)
		{
			Document document;
			document.Parse<0>(args);
			if (document.HasParseError())
			{
				std::cout << "Json Parse error: " << std::endl;
				return 1;
			}

			if (!document.HasMember("img"))
			{
				std::cout << "Json Parse error: " << std::endl;
				return 1;
			}
			Value &img = document["img"];
			if (!img.IsString())
			{
				std::cout << "Json Parse error: " << std::endl;
				return 1;
			}

			std::string imgbase64 = img.GetString();
			int len = calc_data_len(imgbase64.c_str(), imgbase64.length());
			//char * pOut = new char[len + 1];
			memset(m_pBuffer2, 0, BUFFER_SIZE);

			base64_decode(imgbase64.c_str(), imgbase64.length(), m_pBuffer2);
			std::vector<char> data(m_pBuffer2, m_pBuffer2 + len);
			//delete[] pOut;

			cv::Mat image = cv::imdecode(data, CV_LOAD_IMAGE_COLOR);

			std::vector<cv::Rect> rectangles;
			std::vector<float> confidence;
			std::vector<std::vector<cv::Point>> alignment;

			m_body_detector_server->detection_SCALE(image, rectangles, confidence, alignment);

			Document jsondoc;
			jsondoc.SetObject();
			Document::AllocatorType &allocator = jsondoc.GetAllocator();

			Value rectanglesArray(rapidjson::kArrayType);
			Value confidenceArray(rapidjson::kArrayType);
			Value alignmentArray(rapidjson::kArrayType);

			for (int i = 0; i < rectangles.size(); i++)
			{
				std::stringstream stream;

				int x = rectangles[i].x;
				int y = rectangles[i].y;
				int width = rectangles[i].width;
				int height = rectangles[i].height;
				rapidjson::Value object(rapidjson::kObjectType);
				object.AddMember("x", x, allocator);
				object.AddMember("y", y, allocator);
				object.AddMember("width", width, allocator);
				object.AddMember("height", height, allocator);
				rectanglesArray.PushBack(object, allocator);

				std::stringstream stream2;
				stream2 << confidence[i];
				std::string string_temp2 = stream2.str();   //此处也可以用 stream>>string_temp  

				confidenceArray.PushBack(rapidjson::Value((char*)string_temp2.c_str(), allocator), allocator);

				for (int j = 0; j < alignment.size(); j++)
				{
					Value alignmentPtArray(rapidjson::kArrayType);
					for (int m = 0; m < alignment[i].size(); m++)
					{
						Value Pt(rapidjson::kObjectType);
						Pt.AddMember("x", alignment[i][m].x, allocator);
						Pt.AddMember("y", alignment[i][m].y, allocator);
						alignmentPtArray.PushBack(Pt, allocator);
					}
					alignmentArray.PushBack(alignmentPtArray, allocator);
				}
			}
			jsondoc.AddMember("rectangles", rectanglesArray, allocator);    //添加数组
			jsondoc.AddMember("confidence", confidenceArray, allocator);    //添加数组
			jsondoc.AddMember("alignment", alignmentArray, allocator);    //添加数组
			StringBuffer buffer;
			Writer<StringBuffer> writer(buffer);
			jsondoc.Accept(writer);

			strcpy(out,buffer.GetString());
		}
		return 0;
	}
	catch (...)
	{
		std::cout << "unknow exception" << std::endl;
	}
	return -1;
}


int FaceBodyPlugin::GetInterface(
	const std::map<std::string, std::string>& args,
	std::map<std::string, std::string>& out
)
{
	std::cout << "FacePlugin::GetInterface" << std::endl;
	return 1;
}


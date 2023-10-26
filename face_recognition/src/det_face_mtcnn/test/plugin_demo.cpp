#include "det_face_mtcnn.hpp"
#ifdef WIN32
#pragma comment( lib,"winmm.lib" )
#else
typedef unsigned long       DWORD;
#include <time.h> 
#include <dlfcn.h> 
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>
#endif

#include "PluginInterface.h"
#include "PluginMgrInterface.h"
#include "base64.h"
#include "TinyThreadPool.h"



#include "json/rapidjson.h"  
#include "json/document.h"  
#include "json/writer.h"  

#include "File.h"

using namespace rapidjson;
using namespace facethink;
using namespace base::Thread;
using namespace base::base64;
using namespace base::CLock;
using namespace base::File;
using namespace std;
using namespace cv;
using namespace chrono;

typedef int(*pInitfuction)(Plugin::IPluginMgr*, Plugin::IPlugin**, char*, char*); //定义一个函数指针类型
typedef int(*pUninitfuction)(); //定义一个函数指针类型

int testPlugin(int argc, char *argv[])
{

	if (argc < 4) {
		std::cerr << "Usage: " << argv[0]
			<< " det_model"
			<< " image path"
			<< " config file path" << std::endl;
		return 1;
	}

	const std::string det_model = argv[1];
	const std::string image_path = argv[2];
	const std::string config_file_path = argv[3];

#ifdef WIN32
	void* hModule = LoadLibrary("det_face_mtcnn.dll");
#else
	void* hModule = dlopen("./libdet_face_mtcnn.so", RTLD_LAZY);
#endif

	if (!hModule)
	{
#ifdef WIN32
		std::cout << "Error!" << std::endl;
#else
		std::cout << "Error!" << dlerror() << std::endl;
#endif
		return -1;

	}
#ifdef WIN32
	pInitfuction Init = (pInitfuction)GetProcAddress(hModule, "Initialize");
	pUninitfuction Uninit = (pUninitfuction)GetProcAddress(hModule, "UnInitialize");
#else
	pInitfuction Init = (pInitfuction)dlsym(hModule, "Initialize");

	pUninitfuction Uninit = (pUninitfuction)dlsym(hModule, "UnInitialize");

#endif

	if(!Init||!Uninit)
	{
		std::cout << "Get Fuction Error!" << std::endl;
		return -1;
	}
	
	std::cout << det_model << std::endl;
	std::cout << config_file_path << std::endl;

	Plugin::IPlugin* pPlugin = NULL;
	Init(NULL, &pPlugin, (char*)det_model.c_str(),(char*)config_file_path.c_str());
	std::cout << "init succeed" << std::endl;


	std::map<std::string, std::string> inParam;
	std::map<std::string, std::string> outParam;


	std::vector<uchar> buf = FileUtil::readFromFile(image_path.c_str());



	for (int i = 0; i < 1000; i++)
	{

		int base64_len = calc_base64_len(buf.size());
		char * pOut = new char[base64_len+1];
		memset(pOut, 0, sizeof(char)*(base64_len + 1));

		std::string str(buf.begin(), buf.end());
		const char* pData = str.c_str();

		auto start1 = system_clock::now();
		base64_encode(pData, buf.size(), pOut);
		auto end1 = system_clock::now();
		std::cout << "base 64 encode cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << " ms" << std::endl;


		cv::Mat img = cv::imread(image_path); //test image
		if (img.data == 0) {

			delete[] pOut;
			return -1;
		}


		Document doc;
		doc.SetObject();    //key-value 相当与map
							//doc.Setvalue();        //数组型 相当与vector
		Document::AllocatorType &allocator = doc.GetAllocator(); //获取分配器

																 //2，给doc对象赋值
		doc.AddMember("img", rapidjson::Value(pOut, allocator), allocator);
		StringBuffer buffer;
		Writer<StringBuffer> writer(buffer);
		doc.Accept(writer);
		std::string out;	
		char* pszResult = new char[1024 * 80];;
		auto start2 = system_clock::now();
		pPlugin->InvokeMethod("face.detect.multi2", buffer.GetString(), pszResult);
		auto end2 = system_clock::now();
		std::cout << "face.detect.multi2 cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << " ms" << std::endl;
		//std::cout << "result=" << pszResult << std::endl;
		delete[] pOut;
		delete[] pszResult;
	}


	return -1;
}


int main(int argc, char *argv[]){
	testPlugin(argc, argv);
	getchar();
	return 0;
}


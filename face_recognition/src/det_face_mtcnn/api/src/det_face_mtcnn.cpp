#include "frame_state_detector.hpp"
#include "det_face_mtcnn.hpp"
#include "FaceBodyPlugin.h"
#include "PluginMgrInterface.h"
#include <string>
#include "Singleton.h"
using namespace base::CSingleton;

#ifdef WIN32  
#include <Windows.h>  
#else  
#include <stdio.h>  
#include <unistd.h>  
#endif  

namespace facethink {
	using namespace detfacemtcnn;

	FaceDetectionMTCNN::FaceDetectionMTCNN(void) {

	}

	FaceDetectionMTCNN::~FaceDetectionMTCNN(void) {

	}

	FaceDetectionMTCNN* FaceDetectionMTCNN::create(
		const std::string& det1_model_prototxt,
		const std::string& det1_model_binary,
		const std::string& det2_model_prototxt,
		const std::string& det2_model_binary,
		const std::string& det3_model_prototxt,
		const std::string& det3_model_binary,
		const std::string& config_file){

		return new FrameStateDetector(
			det1_model_prototxt,
			det1_model_binary, 
			det2_model_prototxt,
			det2_model_binary,
			det3_model_prototxt,
			det3_model_binary,
		    config_file);
	}

	FaceDetectionMTCNN* FaceDetectionMTCNN::create(
		const std::string& det1_model_file,
		const std::string& det2_model_file,
		const std::string& det3_model_file,
		const std::string& config_file) {

		return new FrameStateDetector(
			det1_model_file,
			det2_model_file,
			det3_model_file,
			config_file);
	}

}

Plugin::IPluginMgr* gPluginMgr = NULL;

std::string getCurrentAppPath()
{
#ifdef WIN32  
	char path[MAX_PATH + 1] = { 0 };
	if (GetModuleFileName(NULL, path, MAX_PATH) != 0)
	{
		std::string strPath(path);
		std::size_t found = strPath.find_last_of("\\");
		std::string retStr = strPath.substr(0, found + 1);
		return retStr;
	}
#else  
	char path[256] = { 0 };
	char filepath[256] = { 0 };
	char cmd[256] = { 0 };
	FILE* fp = NULL;


	sprintf(filepath, "/proc/%d", getpid());

	if (chdir(filepath) != -1)
	{

		snprintf(cmd, 256, "ls -l | grep exe | awk '{print $10}'");
		if ((fp = popen(cmd, "r")) == NULL)
		{
			return std::string();
		}

		if (fgets(path, sizeof(path) / sizeof(path[0]), fp) == NULL)
		{
			pclose(fp);
			return std::string();
		}


		pclose(fp);


		std::string strPath(path);
		std::size_t found = strPath.find_last_of("/");
		std::string retStr = strPath.substr(0, found + 1);
		return retStr;

		//return std::string(path);
	}
#endif  


	return std::string();
}

std::size_t getCpuCount()
{
#ifdef WIN32  
	SYSTEM_INFO sysInfo;
	GetSystemInfo(&sysInfo);
	return sysInfo.dwNumberOfProcessors;
#else  
	long cpu_num = sysconf(_SC_NPROCESSORS_ONLN);
	if (cpu_num == -1)
	{
		assert(false);
		return 0;
	}

	assert(cpu_num == sysconf(_SC_NPROCESSORS_CONF));
	return cpu_num;
#endif  
}


void InitModel(const char*  det_model1, const  char*  det_model2, const  char*  det_model3, const char*  config_file)
{
	std::string str_det_model1;
	std::string str_det_model2;
	std::string str_det_model3;
	std::string str_config_file;
	if(!det_model1|| !det_model2 || !det_model3 || !config_file)
	{
		std::cout << "bin file is empty" << std::endl;
		return;
	}

	str_det_model1 = det_model1;
	str_det_model2 = det_model2;
	str_det_model3 = det_model3;
	str_config_file = config_file;

 	facethink::FaceDetectionMTCNN *body_detector_server = facethink::FaceDetectionMTCNN::create(
		str_det_model1,
		str_det_model2,
		str_det_model3,
 		str_config_file);
	FaceBodyPlugin* pPlugin = CSingleton<FaceBodyPlugin>::getInstance();
	pPlugin->m_body_detector_server = body_detector_server;
	
}

void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
	std::string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (std::string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}


#ifdef WIN32
extern "C" __declspec(dllexport) int  Initialize(
#else
extern "C" __attribute__((visibility("default"))) int  Initialize(
#endif
	Plugin::IPluginMgr* pPluginMgr,
	Plugin::IPlugin** ppPlugin,
	char*  det_model,
	char*  config_file)
{
 	if (pPluginMgr = NULL)
 		return 1 ;	
	if (!det_model || !config_file)
		return 1;
	std::vector<std::string> vecConfig;
	SplitString(std::string(det_model), vecConfig, ",");
	if (vecConfig.size() < 3)
		return 1;
	InitModel(vecConfig[0].c_str(), vecConfig[1].c_str(), vecConfig[2].c_str(), config_file);
	gPluginMgr = pPluginMgr;

 	*ppPlugin = CSingleton<FaceBodyPlugin>::getInstance();


	return 0;
}


#ifdef WIN32
extern "C" __declspec(dllexport) int  UnInitialize()
#else
extern "C" __attribute__((visibility("default"))) int  UnInitialize()
#endif
{
	gPluginMgr = NULL;
	return 0;
}
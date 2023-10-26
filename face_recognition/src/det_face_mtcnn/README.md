# Project: det_face_mtcnn

### 一、Release SDK and Demo:		

		ftp://10.1.2.202/AILab/AI_Engine/det_face_mtcnn

### 二、How to compile it:
 
#### [3rdParty]
You need download third party libs first and save them in the [3rdParty] floder.

		* CUDA9.0		
		* cudnn7.0		
		* platform		
		* rapidjson		
		* boost		
		* ft_caffe
		* openblas
		* opencv
		* protobuf
		* tbb
		* base

#### 1. Windows_GPU
From the main directory,run *cmd* and type:

		* mkdir build		
		* cd build		
		* cmake -G "Visual Studio 14 2015 Win64" ..
		* devenv face_detection_sdk.sln /Build Release

#### 2. Windows_CPU
From the main directory,run *cmd* and type:

		* mkdir build		
		* cd build		
		* cmake -G "Visual Studio 14 2015 Win64" -DCPU_ONLY=ON ..		
		* devenv face_detection_sdk.sln /Build Release		


#### 3. Linux_GPU
	From the main directory,run *cmd* and type:

			* mkdir build		
			* cd build		
			* cmake -DCMAKE_BUILD_TYPE=Release ..
			* make		

#### 4. Linux_CPU
	From the main directory,run *cmd* and type:

			* mkdir build		
			* cd build		
			* cmake -DCMAKE_BUILD_TYPE=Release -DCPU_ONLY=ON ..		
			* make			


Attention:		

		(1)You need install the Visual Studio first;
		(2)"Visual Studio 14 2015" is based on your own Visual Studio Version;
		(3)You must set Win64 in cmake;
		(4)You may need to set the devenv path(Microsoft Visual Studio 14.0\Common7\IDE) in system environment variable;



### 三、How to run the demo and performance_testing:	

##### [Windows]		
From the main directory,run:		

		script\performance_run.bat


### 四、How to use it in your own project:

Please refer to the: [demo](./test/demo.cpp)


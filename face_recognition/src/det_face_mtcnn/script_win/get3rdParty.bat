@echo off

echo download 3rd party
cd ..
cd ..
mkdir 3rdParty
cd 3rdParty

if exist platform (
    echo update platform
    cd platform  
	git pull
	cd ..
) else (
    echo clone platform
    git clone http://gensheng.ye@10.1.2.202/AI_Engine/3rdParty/platform.git
)

if exist ft_caffe (
    echo update ft_caffe
    cd ft_caffe  
	git pull
	cd ..
) else (
    echo clone ft_caffe
    git clone http://gensheng.ye@10.1.2.202/AI_Engine/3rdParty/ft_caffe.git
)
if exist boost (
    echo update boost
    cd boost  
	git pull
	cd ..
) else (
    echo clone boost
    git clone http://gensheng.ye@10.1.2.202/AI_Engine/3rdParty/boost.git
)
if exist openblas (
    echo update openblas
    cd openblas  
	git pull
	cd ..
) else (
    echo clone openblas
    git clone http://gensheng.ye@10.1.2.202/AI_Engine/3rdParty/openblas.git
)
if exist opencv (
    echo update opencv
    cd opencv  
	git pull
	cd ..
) else (
    echo clone opencv
    git clone http://gensheng.ye@10.1.2.202/AI_Engine/3rdParty/opencv.git
)

if exist protobuf (
    echo update protobuf
    cd protobuf  
	git pull
	cd ..
) else (
    echo clone protobuf
    git clone http://gensheng.ye@10.1.2.202/AI_Engine/3rdParty/protobuf.git
)

if exist tbb (
    echo update tbb
    cd tbb  
	git pull
	cd ..
) else (
    echo clone tbb
    git clone http://gensheng.ye@10.1.2.202/AI_Engine/3rdParty/tbb.git
)

if exist rapidjson (
    echo update rapidjson
    cd rapidjson  
	git pull
	cd ..
) else (
    echo clone rapidjson
    git clone http://gensheng.ye@10.1.2.202/AI_Engine/3rdParty/rapidjson.git
)

if exist base (
    echo update base
    cd base  
	git pull
	cd ..
) else (
    echo clone base
    git clone http://gensheng.ye@10.1.2.202/AI_Engine/3rdParty/base.git
)


::pause()
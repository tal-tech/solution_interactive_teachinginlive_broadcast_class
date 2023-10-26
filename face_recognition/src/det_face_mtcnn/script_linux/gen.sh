#!/bin/bash
echo "========== begin to build =========="


### read manifest.ini ###
configFile="../manifest.ini"
function ReadINIfile()  
{   
	Key=$1
	Section=$2
  	Configfile=$3
	ReadINI=`awk -F '=' '/\['$Section'\]/{a=1}a==1&&$1~/'$Key'/{print $2;exit}' $Configfile`  
 	echo "$ReadINI"  
}
sdk_id=`ReadINIfile "id" "main" "$configFile"`
sdk_version=`ReadINIfile "version" "main" "$configFile"`
sdk_model=`ReadINIfile "data" "main" "$configFile"`                                                                  
sdk_model1=${sdk_model:0:27}
sdk_model2=${sdk_model:28:27}
sdk_model3=${sdk_model:56:27}

### sdk_version_cpu ###
array=(${sdk_version//./ })
count=0
for var in ${array[@]}
do
   if [ $count == 0 ] 
   then
       tt=$var
   elif [ $count == 1 ] 
   then
        tt=${tt}".""00"
   else
        tt=${tt}"."${var}
   fi
   count=$[count+1]
done
sdk_version_cpu="$tt"
echo $sdk_version_cpu

### download 3rdParty ###
cd ../..
if [ -d "/3rdParty/" ];then
	echo "exit 3rdParty"
else
	mkdir 3rdParty
fi
cd 3rdParty

for name in platform ft_caffe rapidjson base
do
    if [ -d "/$name/" ];then
	cd $name
	git pull
	cd ..
else
	git clone http://ai_engine:tal2018@10.1.2.202/AI_Engine/3rdParty/$name.git
fi
done


### build ###
cd ../$sdk_id

rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DVERSION="$sdk_version" .. || { echo "cmake $name failed"; exit 1; }
make -j8  || { echo "make $name failed"; exit 1; }
cd ..

rm -rf build_cpu
mkdir build_cpu
cd build_cpu
cmake -DCMAKE_BUILD_TYPE=Release -DCPU_ONLY=YES -DVERSION="$sdk_version_cpu" .. || { echo "cmake $name failed"; exit 1; }
make -j8  || { echo "make $name failed"; exit 1; }
cd ..

### release copy filse ###
mkdir -p release/${sdk_id}"_"${sdk_version}/libs/linux/gpu
cp -r build/bin/lib* release/${sdk_id}"_"${sdk_version}/libs/linux/gpu
cp build/bin/performance_testing_GPU release/${sdk_id}"_"${sdk_version}/libs/linux/gpu

mkdir -p release/${sdk_id}"_"${sdk_version}/libs/linux/cpu
cp -r build_cpu/bin/lib* release/${sdk_id}"_"${sdk_version}/libs/linux/cpu
cp build_cpu/bin/performance_testing_CPU release/${sdk_id}"_"${sdk_version}/libs/linux/cpu

mkdir -p release/${sdk_id}"_"${sdk_version}/demo
cp -r test/demo.cpp release/${sdk_id}"_"${sdk_version}/demo

mkdir -p release/${sdk_id}"_"${sdk_version}/include
cp -r api/include/${sdk_id}".hpp" release/${sdk_id}"_"${sdk_version}/include

mkdir -p release/${sdk_id}"_"${sdk_version}/model
cp -r data/models/$sdk_model1 release/${sdk_id}"_"${sdk_version}/model
cp -r data/models/$sdk_model2 release/${sdk_id}"_"${sdk_version}/model
cp -r data/models/$sdk_model3 release/${sdk_id}"_"${sdk_version}/model
cp -r data/config.ini release/${sdk_id}"_"${sdk_version}/model

mkdir -p release/${sdk_id}"_"${sdk_version}/images
cp -r data/images/testing release/${sdk_id}"_"${sdk_version}/images

cp manifest.ini release/${sdk_id}"_"${sdk_version}
cp script_linux/performance_run_cpu.sh release/${sdk_id}"_"${sdk_version}
cp script_linux/performance_run_gpu.sh release/${sdk_id}"_"${sdk_version}
 
cp ../3rdParty/ft_caffe/cuda9/libs/Linux/libfacethink_caffe_GPU* release/${sdk_id}"_"${sdk_version}/libs/linux/gpu
cp ../3rdParty/ft_caffe/cuda9/libs/Linux/libfacethink_caffe_CPU* release/${sdk_id}"_"${sdk_version}/libs/linux/cpu

cp ../3rdParty/platform/lib/Linux/lib* release/${sdk_id}"_"${sdk_version}/libs/linux/gpu
cp ../3rdParty/platform/lib/Linux/lib* release/${sdk_id}"_"${sdk_version}/libs/linux/cpu


echo "========================================================"
echo "==========Congratulations!!! build success=============="
echo "========================================================"


#!/bin/bash
echo "========== compress and upload to ftp =========="

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

### compress to zip ###
cd ../release
zip -r ${sdk_id}"_"${sdk_version}".zip" ${sdk_id}"_"${sdk_version}

mkdir temp
cd temp

### zip if exist ###
ftp -v -n 10.1.2.202<<EOF
user webmaster facethink2016
binary
cd AILab
cd AI_Engine
mkdir $sdk_id
cd $sdk_id
mkdir release
cd release
mkdir Linux
cd Linux
get ${sdk_id}"_"${sdk_version}".zip"
bye
EOF

if [ -f ${sdk_id}"_"${sdk_version}".zip" ]; then
	rm ${sdk_id}"_"${sdk_version}".zip"
	echo ${sdk_id}"_"${sdk_version}".zip" is already exit
    echo upload ${sdk_id}"_"${sdk_version}".zip" failed!!!
	exit
fi

cd ..

### upload zip to ftp ###
ftp -v -n 10.1.2.202<<EOF
user webmaster facethink2016
binary
cd AILab
cd AI_Engine
mkdir $sdk_id
cd $sdk_id
mkdir release
cd release
mkdir Linux
cd Linux
lcd ./
prompt
put ${sdk_id}"_"${sdk_version}".zip"
bye
EOF

cd ..
rm -rf release
echo "commit to ftp successfully"

@echo off

::1-获取属性信息-manifest.ini
CALL readconfig ..\manifest.ini version vnumber
set "vnumber=%vnumber: =%"
CALL readconfig ..\manifest.ini id curdir
set "curdir=%curdir: =%"

::2-压缩文件夹
set rfloder=%curdir%_%vnumber%
cd ..
cd release
..\script_win\WinRAR.exe a %rfloder%.zip %rfloder% -r

::3-上传文件夹
echo open 10.1.2.202 21>> temp.txt
echo webmaster>> temp.txt
echo facethink2016>> temp.txt
echo cd AILab>> temp.txt
echo cd AI_Engine>> temp.txt
echo mkdir %curdir%>> temp.txt
echo cd %curdir%>> temp.txt
echo mkdir release>> temp.txt
echo cd release>> temp.txt
echo mkdir Win64>> temp.txt
echo cd Win64>> temp.txt

echo bin >> temp.txt
echo put %rfloder%.zip >> temp.txt

echo bye >> temp.txt
ftp -i -s:temp.txt

del /q temp.txt

::pause
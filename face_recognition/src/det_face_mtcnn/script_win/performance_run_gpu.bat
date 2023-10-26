@echo off

::det_face_mtcnn model
SET CAFFE_MODEL1=model/det_face_1_v2.0.0.bin
SET CAFFE_MODEL2=model/det_face_2_v2.0.0.bin
SET CAFFE_MODEL3=model/det_face_3_v2.0.0.bin


::Set test data
SET IMAGES_PATH=images/testing/

SET CONFIG_PATH=model/config.ini

start libs/x64/gpu/performance_testing_GPU.exe ^
%CAFFE_MODEL1% ^
%CAFFE_MODEL2% ^
%CAFFE_MODEL3% ^
%IMAGES_PATH% ^
%CONFIG_PATH%

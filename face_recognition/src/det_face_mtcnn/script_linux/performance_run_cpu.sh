
# Set body box model
CAFFE_MODEL1=model/det_face_1_v2.0.0.bin
CAFFE_MODEL2=model/det_face_2_v2.0.0.bin
CAFFE_MODEL3=model/det_face_3_v2.0.0.bin


# Set test data
IMAGES_PATH=images/testing/

# Set test data
CONFIG_PATH=model/config.ini

libs/linux/cpu/performance_testing_CPU $CAFFE_MODEL1 $CAFFE_MODEL2 $CAFFE_MODEL3 $IMAGES_PATH $CONFIG_PATH


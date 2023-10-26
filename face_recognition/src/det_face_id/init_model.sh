#!/bin/bash

MODEL_NAME=model/det_face_id_v1.0.3-1
AI_MODEL_NAME=/home/guoweiye/workspace/face_recognition/src/det_face_id/model/det_face_id_v1.0.3-1
UBUNTU_CUDA10=/home/guoweiye/workspace/face_recognition/src/3rdParty/ubuntu-cuda10

cd /home/guoweiye/workspace/face_recognition/src/det_face_id

# if trt file not exist
#if [ ! -f "${AI_MODEL_NAME}.trt" ]; then
  GPU=`nvidia-smi  -L`
  echo $GPU 1>&1

  #if [[ $GPU == *"Tesla P4 "* ]]; then
  #  echo 'Find P4' >&1
  #  cp ${MODEL_NAME}.trt-P4  ${AI_MODEL_NAME}.trt
  if [[ $GPU == *"Tesla P40 "* ]]; then
    echo 'Find P40' >&1
    cp ${MODEL_NAME}.trt-P40  ${AI_MODEL_NAME}.trt
  elif [[ $GPU == *"GeForce GTX 745"* ]]; then
    echo 'Find GeForce GTX 745' >&1
    cp ${MODEL_NAME}.trt-GTX745  ${AI_MODEL_NAME}.trt
  else
    echo 'Find new nivdia' >&1
    echo "alertcode:91001009,alertmsg:new nvidia $GPU" >&1

    MODEL=model/det_face_id_v1.0.3-1.uff
    export LD_LIBRARY_PATH=libs/linux/gpu/:/usr/local/cuda/lib64:${UBUNTU_CUDA10}

    libs/linux/gpu/performance_testing_GPU_for_trt $MODEL 1 1 1 1
    cp ${MODEL_NAME}.trt  ${AI_MODEL_NAME}.trt
  fi
#fi

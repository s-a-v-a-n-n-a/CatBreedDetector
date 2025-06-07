#!/usr/bin/bash

ONNX_PATH=$1
MODEL_NAME=$2
MODEL_PATH=$3
MODEL_VERSION=$4

docker run --gpus=all -it --rm \
    -v $PWD:/workspace \
    nvcr.io/nvidia/tensorrt:23.10-py3 \
    trtexec --onnx=$ONNX_PATH --saveEngine=$MODEL_PATH/$MODEL_NAME/$MODEL_VERSION/model.plan --inputIOFormats=fp32:chw

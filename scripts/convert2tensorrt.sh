#!/usr/bin/bash

ONNX_PATH=$1
OUTPUT_DIR=$2

echo "Converting $ONNX_PATH to TensorRT ($PRECISION)..."

mkdir -p $OUTPUT_DIR

/usr/src/tensorrt/bin/trtexec --onnx=$ONNX_PATH --saveEngine=$OUTPUT_DIR/model.fp32.trt --inputIOFormats=fp32:chw

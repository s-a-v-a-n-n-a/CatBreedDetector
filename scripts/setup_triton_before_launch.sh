#!/usr/bin/bash

ONNX_PATH=$1
MODEL_VERSION=$2
IMAGES_PATH=$3

echo "Getting onnx model..."
dvc pull $ONNX_PATH

echo "Converting onnx model to TensorRT model of suitable version..."
scripts/convert2tensorrt_8.6.sh $ONNX_PATH cat_breed_detector triton/models $MODEL_VERSION

echo "Copying images..."
mkdir triton/images
dvc pull $IMAGES_PATH
cp -a $IMAGES_PATH/. triton/images/

cd triton
docker compose -p $NETWORK_NAME up -d

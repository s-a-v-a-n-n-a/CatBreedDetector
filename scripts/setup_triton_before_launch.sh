#!/usr/bin/bash

ONNX_PATH=$1
MODEL_VERSION=$2
IMAGES_PATH=$3

echo "Getting onnx model..."
if [ ! -e $ONNX_PATH ]; then
    dvc pull $ONNX_PATH
fi

mkdir triton/models/cat_breed_detector/$MODEL_VERSION
echo "Converting onnx model to TensorRT model of suitable version..."
bash scripts/convert2tensorrt_8.6.sh $ONNX_PATH cat_breed_detector triton/models $MODEL_VERSION

echo "Copying images..."
mkdir triton/images
if [ ! -e $IMAGES_PATH ]; then
    dvc pull $IMAGES_PATH
fi
cp -a $IMAGES_PATH/. triton/images/

cd triton
docker compose -p $NETWORK_NAME up -d

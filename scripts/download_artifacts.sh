#!/usr/bin/bash

ROOT="$(dirname "$(dirname "$(readlink -fm "$0")")")"
ARTIFACTS_PATH="$ROOT/data/artifacts"
mkdir $ARTIFACTS_PATH
cd $ARTIFACTS_PATH

wget 'https://cdn.shopify.com/s/files/1/2668/1922/files/british-shorthair-1.jpg' -O british_cat.jpg
wget 'https://www.thesprucepets.com/thmb/aWULXjTWxZbCJ4GixA7JMw8K15w=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/GettyImages-1189893683-e0ff70596b3b4f0687ba573e5a671f74.jpg' -O maincoon_cat.jpg
wget 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQOs9yJ-Djk40g7UA8MqPQW99CyZKSovPd2Sw&s' -O angora.jpg
wget 'https://news.cvm.ncsu.edu/wp-content/uploads/sites/3/2017/04/siamemsescats-e1491411839658.jpg' -O siamese_cat.jpg

dvc add $ARTIFACTS_PATH
dvc push

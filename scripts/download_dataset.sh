#!/usr/bin/bash
ROOT="$(dirname "$(dirname "$(readlink -fm "$0")")")"
DATA_PATH="$ROOT/data/data"
python3 ../cat_breed_detector/download_data.py --url ma7555/cat-breeds-dataset --target_dir $DATA_PATH

dvc add $DATA_PATH
dvc push

#!/usr/bin/bash
ROOT="$(dirname "$(dirname "$(readlink -fm "$0")")")"
DATA_PATH="$ROOT/data/data_refined"
python3 ../cat_breed_detector/download_data.py --url doctrinek/catbreedsrefined-7k --target_dir $DATA_PATH

dvc add $DATA_PATH
dvc push

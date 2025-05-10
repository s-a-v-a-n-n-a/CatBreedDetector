#!/usr/bin/bash
ROOT="$(dirname "$(dirname "$(readlink -fm "$0")")")"
DATA_PATH = "$ROOT/data"
python3 download_dataset.py --url ma7555/cat-breeds-dataset --target_dir DATA_PATH

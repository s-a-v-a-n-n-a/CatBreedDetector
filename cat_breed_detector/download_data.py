import os
import shutil
from pathlib import Path

import fire
import kagglehub

from dvc.api import DVCFileSystem


def download(url: str, target_dir: str) -> None:
    path = kagglehub.dataset_download(url)
    print("Downloaded files to path:", path)

    shutil.copytree(path, target_dir, dirs_exist_ok=True)
    print("Moved files to path:", target_dir)

    shutil.rmtree(path)
    print("Removed files from ", path)


def ensure_data_downloaded(data_path: Path):
    fs = DVCFileSystem("./data")
    if not os.path.exists(data_path):
        fs.get(data_path, data_path, recursive=True)


if __name__ == "__main__":
    fire.Fire(download)

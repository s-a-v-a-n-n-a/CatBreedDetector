import shutil

import fire
import kagglehub


def download(url: str, target_dir: str) -> None:
    path = kagglehub.dataset_download(url)
    print("Downloaded files to path:", path)

    shutil.copytree(path, target_dir, dirs_exist_ok=True)
    print("Moved files to path:", target_dir)

    shutil.rmtree(path)
    print("Removed files from ", path)


if __name__ == "__main__":
    fire.Fire(download)

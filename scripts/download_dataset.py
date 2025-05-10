import os
import fire
import shutil
import kagglehub


def download(
        url: str,
        target_dir: str
) -> None:
    path = kagglehub.dataset_download(url)
    print("Downloaded files to path:", path)

    result_data_path = os.path.join(
        target_dir,
        os.path.basename(path)
    )
    shutil.copytree(
        path,
        result_data_path
    )
    print("Moved files to path:", result_data_path)

    shutil.rmtree(path)
    print("Removed files from ", result_data_path)


if __name__ == "__main__":
    fire.Fire(download)

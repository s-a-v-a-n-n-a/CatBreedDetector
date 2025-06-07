from pathlib import Path

from dvc.api import DVCFileSystem


def ensure_data_unpacked(data_path: Path):
    fs = DVCFileSystem("./data")
    if not Path(data_path).exists():
        print(f"Getting {data_path} from dvc")
        fs.get(data_path, data_path, recursive=True)

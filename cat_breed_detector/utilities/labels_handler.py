import json
from pathlib import Path


def get_labels_metainfo(labels_meta_path: Path) -> dict[str, any]:
    with open(labels_meta_path, "r") as labels_meta_file:
        labels_meta = json.load(labels_meta_file)
        return labels_meta


def write_labels_metainfo(labels_meta_path: Path, labels_meta: dict[str, any]) -> None:
    Path(labels_meta_path).parent.mkdir(exist_ok=True)
    with open(labels_meta_path, "w") as labels_meta_file:
        json.dump(labels_meta, labels_meta_file, indent=4)

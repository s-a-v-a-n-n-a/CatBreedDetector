import typing as tp
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from dvc.api import DVCFileSystem
from PIL import Image
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor

import utilities.labels_handler as labels_handler
from utilities.model_getter import get_processor


class InferDataset(torch.utils.data.Dataset):
    def __init__(
        self, path: str, processor: ViTImageProcessor, transform: tp.Any = None
    ) -> None:
        file_names = []
        for file in sorted(Path(path).glob("**/*")):
            if str(file).endswith(".jpg") or str(file).endswith(".png"):
                file_names.append(str(file))
        self.df = pd.DataFrame.from_dict({"image": file_names})
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        img_path = self.df.iloc[idx]["image"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exception_msg:
            print(f"Not worked on path {img_path}: {exception_msg}")
            raise exception_msg
        if self.transform:
            images = self.transform(images=image)
        else:
            images = self.processor(images=image, return_tensors="pt")

        return {"pixel_values": images["pixel_values"][0].clone().detach()}


class InferDataModule(pl.LightningDataModule):
    _processor = get_processor()
    _image_mean, _image_std = _processor.image_mean, _processor.image_std
    _size = _processor.size["height"]

    def __init__(self, config: dict[str, any], infer_data_path: Path) -> None:
        super().__init__()
        self._data_dir = infer_data_path
        self._config = config
        self.batch_size = config["training"]["batch_size"]
        self.num_workers = config["training"]["num_workers"]

        self._seed = config["training"]["seed"]
        self._generator = torch.Generator().manual_seed(config["training"]["seed"])

    @staticmethod
    def ensure_data_downloaded(data_path: Path):
        fs = DVCFileSystem("./data")
        if not Path(data_path).exists():
            fs.get(data_path, data_path, recursive=True)

    def prepare_data(self):
        self.ensure_data_downloaded(self._data_dir)
        if not Path(self._data_dir).is_dir():
            raise Exception("For inference a directory should be passed")
        self.ensure_data_downloaded(self._config["data_loading"]["labels2id_meta"])
        self.ensure_data_downloaded(self._config["data_loading"]["id2labels_meta"])

    def setup(self, stage: tp.Optional[str] = None):
        self._num_labels = self._config["model"]["num_labels"]
        self.dataset = InferDataset(self._data_dir, processor=self._processor)
        self.id2label = labels_handler.get_labels_metainfo(
            self._config["data_loading"]["id2labels_meta"]
        )
        self.label2id = labels_handler.get_labels_metainfo(
            self._config["data_loading"]["labels2id_meta"]
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.dataset, batch_size=1, shuffle=False, num_workers=self.num_workers
        )

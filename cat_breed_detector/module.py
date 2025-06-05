import json
import torch
from pathlib import Path
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from datasets import Dataset, Image, ClassLabel
from model_getter import get_model, get_processor, MODEL_SOURCE
from dataset import CatBreedDataset
from torch.nn.utils.rnn import pad_sequence


class CustomDataModule(pl.LightningDataModule):

    _processor = get_processor()
    _image_mean, _image_std = _processor.image_mean, _processor.image_std
    _size = _processor.size["height"]
    _train_transforms = transforms.Compose([
        transforms.Resize((_size, _size)),
        transforms.RandomRotation(90),
        transforms.RandomAdjustSharpness(2),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=_image_mean, std=_image_std)
    ])
    _val_transforms = transforms.Compose([
        transforms.Resize((_size, _size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_image_mean, std=_image_std)
    ])

    def __init__(
            self,
            data_dir: Path,
            num_labels: int,
            id2label_meta_file: Path,
            label2id_meta_file: Path,
            batch_size: int,
            num_workers: int,
            data_split_ratio: float,
            seed: int
    ) -> None:
        super().__init__()
        self._data_dir = data_dir
        self._num_labels = num_labels
        self.id2label_meta_file = id2label_meta_file
        self.label2id_meta_file = label2id_meta_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_split_ratio = data_split_ratio

        self._seed = seed
        self._generator = torch.Generator().manual_seed(seed)

    @staticmethod
    def train_transforms(examples: dict[str, any]) -> dict[str, any]:
        examples['pixel_values'] = [
            CustomDataModule._train_transforms(image.convert("RGB"))
            for image in examples['images']
        ]
        return examples

    @staticmethod
    def val_transforms(examples: dict[str, any]) -> dict[str, any]:
        examples['pixel_values'] = [
            CustomDataModule._val_transforms(image.convert("RGB"))
            for image in examples['images']
        ]
        return examples

    @staticmethod
    def collate_fn(examples: dict[str, any]) -> dict[str, any]:
        pixel_values = pad_sequence(
            [example["pixel_values"]for example in examples],
            batch_first=True
        )

        labels = torch.stack([example['labels'] for example in examples])

        return {"pixel_values": pixel_values, "labels": labels}

    def train_test_split(
            self,
            dataset: Dataset
    ) -> tuple[CatBreedDataset, CatBreedDataset]:
        total_size = len(dataset)
        train_ratio = self.data_split_ratio
        train_size = int(train_ratio * total_size)
        test_size = total_size - train_size

        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=self._generator
        )
        custom_train_dataset = CatBreedDataset.from_dataset(
            train_dataset,
            processor=get_processor(),
            transform=CustomDataModule.train_transforms,
        )
        custom_test_dataset = CatBreedDataset.from_dataset(
            test_dataset,
            processor=get_processor(),
            transform=CustomDataModule.val_transforms,
        )
        return custom_train_dataset, custom_test_dataset

    def train_val_split(
            self,
            dataset: Dataset
    ) -> tuple[CatBreedDataset, CatBreedDataset]:
        total_size = len(dataset)
        train_ratio = self.data_split_ratio
        train_size = int(train_ratio * total_size)
        val_size = total_size - train_size

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=self._generator
        )
        custom_train_dataset = CatBreedDataset.from_dataset(
            train_dataset,
            processor=get_processor(),
            transform=CustomDataModule.train_transforms,
        )
        custom_val_dataset = CatBreedDataset.from_dataset(
            val_dataset,
            processor=get_processor(),
            transform=CustomDataModule.val_transforms,
        )
        return custom_train_dataset, custom_val_dataset

    @staticmethod
    def get_labels_metainfo(
            labels_meta_path: Path
    ) -> dict[str, any]:
        with open(labels_meta_path, "r") as labels_meta_file:
            labels_meta = json.load(labels_meta_file)
            return labels_meta

    @staticmethod
    def write_labels_metainfo(
            labels_meta_path: Path,
            labels_meta: dict[str, any]
    ) -> None:
        with open(labels_meta_path, "w") as labels_meta_file:
            json.dump(labels_meta, labels_meta_file, indent=4)

    def setup(self, stage=None):
        self.dataset = CatBreedDataset.from_data_path(
            self._data_dir,
            self._num_labels,
            processor=get_processor(),
            randomize=True,
            seed=self._seed,
            transform=CustomDataModule.val_transforms,
        )
        self.labels = sorted(list(set(self.dataset.labels)))
        print(len(self.labels))
        self.id2label = self.dataset.id2label
        self.write_labels_metainfo(
            self.id2label_meta_file,
            self.id2label
        )
        self.label2id = self.dataset.label2id
        self.write_labels_metainfo(
            self.label2id_meta_file,
            self.label2id
        )

        self.custom_train_dataset, self.custom_test_dataset = self.train_test_split(
            self.dataset.dataset
        )
        self.custom_train_dataset, self.custom_val_dataset = self.train_val_split(
            self.custom_train_dataset.dataset
        )

    def train_dataloader(self):
        return DataLoader(
            self.custom_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=CustomDataModule.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.custom_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=CustomDataModule.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.custom_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=CustomDataModule.collate_fn
        )

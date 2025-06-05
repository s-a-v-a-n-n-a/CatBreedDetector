from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import ClassLabel, Dataset as HFDataset
import PIL.Image as Image
from torch.utils.data import WeightedRandomSampler
import torchvision.transforms as transforms
from collections import Counter
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification
)


class CatBreedDataset(Dataset):
    def __init__(
            self,
            dataset: Dataset,
            processor: ViTImageProcessor,
            transform: transforms.Compose = None
    ) -> None:
        self._dataset = dataset

        self.processor = processor
        self.transform = transform

    @classmethod
    def from_dataset(
            cls,
            dataset: Dataset,
            processor: ViTImageProcessor,
            transform: transforms.Compose = None
    ) -> 'CatBreedDataset':
        result = cls(
            dataset,
            processor,
            transform
        )
        result.df = pd.DataFrame()
        batch_size = 1000
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            if isinstance(batch, dict):
                batch_df = pd.DataFrame.from_dict(batch)
            else:
                batch_df = batch.to_pandas()
            result.df = pd.concat([result.df, batch_df])
        result.labels = sorted(list(set([element["label"] for element in tqdm(dataset)])))
        result.label2id = {label: i for i, label in enumerate(result.labels)}
        result.id2label = {i: label for i, label in enumerate(result.labels)}
        return result

    @classmethod
    def from_data_path(
            cls,
            data_dirs: list[Path],
            num_labels: int,
            processor: ViTImageProcessor,
            seed: int,
            transform: transforms.Compose,
            randomize: bool = True,
    ) -> 'CatBreedDataset':
        df, labels = CatBreedDataset.get_dataframe_from_dir(
            data_dirs,
            num_labels
        )
        result = cls(
            CatBreedDataset.get_dataset_from_dataframe(
                df,
                labels,
                randomize,
                seed
            ),
            processor,
            transform
        )
        result.df = df
        result.labels = labels
        result.label2id = {label: i for i, label in enumerate(result.labels)}
        result.id2label = {i: label for i, label in enumerate(result.labels)}
        return result

    @property
    def labels_list(self):
        return self.labels

    @property
    def dataset(self):
        return self._dataset

    @staticmethod
    def get_dataframe_from_dir(
            dataset_paths: list[Path],
            num_labels: int
    ) -> pd.DataFrame:
        file_names = []
        labels = []

        for dataset_path in dataset_paths:
            for file in tqdm(sorted(Path(dataset_path).glob('*/*.*'))):
                if str(file).endswith('.jpg') or str(file).endswith('.png'): 
                    file_names.append(str(file))
                    label = str(file).split('/')[-2]
                    labels.append(label)
        df = pd.DataFrame.from_dict({"image": file_names, "label": labels})

        df = df[df['label'].isin(df['label'].value_counts().head(num_labels).index)]
        labels = sorted(list(set(df["label"])))
        print(len(labels))
        redundant_labels_dict = {
            'Norwegian Forest Cat': 'Norwegian Forest',
            'Sphynx - Hairless Cat': 'Sphynx'
        }
        df['label'] = df['label'].replace(redundant_labels_dict)

        labels = sorted(list(set(df["label"])))
        print(len(labels))

        return df, labels

    @staticmethod
    def get_dataset_from_dataframe(
            df,
            labels_list,
            randomize: bool,
            seed: int
    ):
        ClassLabels = ClassLabel(
            num_classes=len(labels_list),
            names=labels_list
        )

        if randomize:
            chunks = np.array_split(df, max(1, len(df)//10000))
            processed_chunks = []
            
            for i, chunk in enumerate(chunks):
                processed_chunk = CatBreedDataset.random_resample(
                    chunk,
                    seed
                )
                processed_chunks.append(processed_chunk)
            
            resampled_df = pd.concat(processed_chunks, ignore_index=True)
        else:
            resampled_df = df
        dataset = HFDataset.from_pandas(
            resampled_df
        )
        def map_label2id(example):
            example['label'] = ClassLabels.str2int(example['label'])
            return example

        dataset = dataset.map(map_label2id, batched=True)
        dataset = dataset.cast_column('label', ClassLabels)

        return dataset

    @staticmethod
    def random_resample(
            dataframe: pd.DataFrame,
            seed: int = 42
    ) -> pd.DataFrame:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if 'label' not in dataframe.columns:
            raise ValueError("DataFrame must contain 'label' column")

        unique_labels = dataframe['label'].unique()
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        labels = dataframe['label'].map(label_to_idx).to_numpy()

        class_counts = Counter(labels)
        if len(class_counts) < 2:
            return dataframe
        
        max_count = max(class_counts.values())
        second_largest = sorted(class_counts.values())[-2]

        class_indices = {label: np.where(labels == label)[0] for label in class_counts}
        
        undersampled_indices = []
        for label, indices in class_indices.items():
            if class_counts[label] == max_count:
                np.random.seed(seed)
                selected = np.random.choice(indices, size=second_largest, replace=False)
                undersampled_indices.append(selected)
            else:
                undersampled_indices.append(indices)
        
        undersampled_indices = np.concatenate(undersampled_indices)

        label_counts = np.array([class_counts[label] for label in labels[undersampled_indices]])
        sample_weights = 1. / label_counts
        sample_weights /= sample_weights.sum()
        
        np.random.seed(seed)
        resampled_indices = np.random.choice(
            undersampled_indices,
            size=len(undersampled_indices),
            replace=True,
            p=sample_weights
        )

        resampled_df = dataframe.iloc[resampled_indices].copy()
        
        return resampled_df

    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image']
        got_label = self.df.iloc[idx]['label']
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"Not worked on path {img_path}")
        label_id = self.label2id[got_label]
        result = {
            "images": [image],
            "labels": [label_id]
        }
        if self.transform:
            images = self.transform(result)
        else:
            images = self.processor(
                result, return_tensors="pt"
            )['pixel_values'][0]
            
        return {
            'pixel_values': images['pixel_values'][0].clone().detach(),
            'labels': torch.tensor(label_id).clone().detach()
        }

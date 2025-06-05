import torch
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification
)
import pytorch_lightning as pl
MODEL_SOURCE = 'google/vit-base-patch16-224-in21k'


def get_model(
        datamodule: pl.LightningDataModule
) -> torch.nn.Module:
    """Get model to work with"""
    
    model = ViTForImageClassification.from_pretrained(
        MODEL_SOURCE,
        num_labels=len(datamodule.id2label),
        id2label=datamodule.id2label,
        label2id=datamodule.label2id
    )
    return model


def get_processor():
    processor = ViTImageProcessor.from_pretrained(
        MODEL_SOURCE
    )
    return processor

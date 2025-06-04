import torch
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification
)
MODEL_SOURCE = 'google/vit-base-patch16-224-in21k'


def get_model(
    num_labels: int,
    id2label: dict,
    label2id: dict
) -> torch.nn.Module:
    """Get model to work with"""
    
    model = ViTForImageClassification.from_pretrained(
        MODEL_SOURCE,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    return model


def get_processor():
    processor = ViTImageProcessor.from_pretrained(
        MODEL_SOURCE
    )
    return processor

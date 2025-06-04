import hydra
import torch
import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from datasets import Dataset, Image, ClassLabel

from logging_selector import get_logger
from model_getter import get_model, get_processor, MODEL_SOURCE
from module import CustomDataModule
from trainer import CatBreedClassifier

from dataset import CatBreedDataset

from download_data import ensure_data_downloaded


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    ensure_data_downloaded(config["data_loading"]["train_data_path"])
    ensure_data_downloaded(config["data_loading"]["labels2id_meta"])
    ensure_data_downloaded(config["data_loading"]["id2labels_meta"])
    datamodule = CustomDataModule(
        config["data_loading"]["train_data_path"],
        config["model"]["num_labels"],
        config["data_loading"]["labels2id_meta"],
        config["data_loading"]["id2labels_meta"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        train_val_ratio=config["training"]["train_val_ratio"],
        seed=config["training"]["seed"]
    )
    datamodule.train_setup()

    logger = get_logger(config["logging"])

    labels_list = datamodule.labels
    id2label = datamodule.id2label
    label2id = datamodule.label2id
    assert len(labels_list) == config["model"]["num_labels"]
    model = get_model(
        len(labels_list),
        id2label,
        label2id
    )
    model_name = MODEL_SOURCE
    num_epochs = config["training"]["num_epochs"]
    num_training_steps = num_epochs * len(datamodule.train_dataloader())
    num_warmup_steps = int(
        config["training"]["num_warmup_steps_ratio"] * num_training_steps
    )
    module = CatBreedClassifier(
        model,
        num_labels=len(labels_list),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        datamodule=datamodule
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=config["model"]["model_local_path"],
        filename="model_{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        log_every_n_steps=config["logging"]["steps_to_log"],
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()

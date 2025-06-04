import os
import fire
import hydra
from hydra import compose, initialize
import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from model_getter import get_model, get_processor, MODEL_SOURCE
from module import CustomDataModule
from trainer import CatBreedClassifier

from download_data import ensure_data_downloaded


def main(test_dir: str, checkpoint_name: str) -> None:
    with initialize(config_path="../configs"):
        config = compose(config_name="config")

    ensure_data_downloaded(config["data_loading"]["test_data_path"])
    ensure_data_downloaded(config["data_loading"]["labels2id_meta"])
    ensure_data_downloaded(config["data_loading"]["id2labels_meta"])
    datamodule = CustomDataModule(
        config["data_loading"]["test_data_path"],
        config["model"]["num_labels"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        train_val_ratio=config["training"]["train_val_ratio"],
        seed=config["training"]["seed"],
        id2label_meta_file=config["data_loading"]["id2labels_meta"],
        label2id_meta_file=config["data_loading"]["labels2id_meta"]
    )
    datamodule.prepare_data()
    datamodule.test_setup()

    id2label = CustomDataModule.get_labels_metainfo(config["data_loading"]["id2labels_meta"])
    label2id = CustomDataModule.get_labels_metainfo(config["data_loading"]["labels2id_meta"])
    model = get_model(
        config["model"]["num_labels"],
        id2label,
        label2id
    )
    model_name = MODEL_SOURCE

    module = CatBreedClassifier.load_from_checkpoint(
        f"{config['model']['model_local_path']}/{checkpoint_name}",
        model=model,
        lr=config["training"]["lr"],
        for_test=True
    )
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        log_every_n_steps=config["logging"]["test_steps_to_log"],
    )

    results = trainer.test(
        module,
        dataloaders=datamodule.test_dataloader()
    )
    print(results)


if __name__ == "__main__":
    fire.Fire(main)

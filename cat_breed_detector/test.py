import os
import fire
import hydra
from hydra import compose, initialize
import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from logging_selector import get_logger
from model_getter import get_model, get_processor, MODEL_SOURCE
from module import CustomDataModule
from trainer import CatBreedClassifier

from download_data import ensure_data_downloaded


def main(test_dir: str, checkpoint_name: str) -> None:
    with initialize(config_path="../configs"):
        config = compose(config_name="config")

        data_directories = config["data_loading"]["data_path"]
        for data_path in data_directories:
            ensure_data_downloaded(data_path)
        ensure_data_downloaded(config["data_loading"]["labels2id_meta"])
        ensure_data_downloaded(config["data_loading"]["id2labels_meta"])
        datamodule = CustomDataModule(
            data_directories,
            config["model"]["num_labels"],
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"]["num_workers"],
            data_split_ratio=config["training"]["train_val_ratio"],
            seed=config["training"]["seed"],
            id2label_meta_file=config["data_loading"]["id2labels_meta"],
            label2id_meta_file=config["data_loading"]["labels2id_meta"]
        )
        datamodule.prepare_data()
        datamodule.setup()
        logger = get_logger(config["logging"])

        labels_list = datamodule.labels
        id2label = datamodule.id2label
        label2id = datamodule.label2id
        model = get_model(
            len(labels_list),
            id2label,
            label2id
        )
        model_name = MODEL_SOURCE

        module = CatBreedClassifier.load_from_checkpoint(
            f"{config['model']['model_local_path']}/{checkpoint_name}",
            model=model,
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            for_test=True
        )
        trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            logger=logger,
            log_every_n_steps=config["logging"]["test_steps_to_log"],
        )

        results = trainer.test(
            module,
            dataloaders=datamodule.test_dataloader()
        )
        print(results)


if __name__ == "__main__":
    fire.Fire(main)

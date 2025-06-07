import fire
import pytorch_lightning as pl
import torch
from hydra import compose, initialize

from modules.module import Module
from modules.vit_data_module import ViTDataModule
from utilities.data_handler import ensure_data_unpacked
from utilities.logging_selector import get_logger
from utilities.model_getter import get_model


def main(test_dir: str, checkpoint_name: str) -> None:
    with initialize(config_path="../configs", version_base="1.1"):
        config = compose(config_name="config")

        datamodule = ViTDataModule(config)
        datamodule.prepare_data()
        datamodule.setup()
        logger = get_logger(config["logging"])

        model = get_model(datamodule)
        torch.set_float32_matmul_precision("medium")

        ensure_data_unpacked(f"{test_dir}/{checkpoint_name}")
        module = Module.load_from_checkpoint(
            f"{test_dir}/{checkpoint_name}",
            model=model,
            datamodule=datamodule,
            train_parameters=config["training"],
            log_parameters=config["logging"],
            for_test=True,
        )
        trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            logger=logger,
            log_every_n_steps=config["logging"]["test_steps_to_log"],
        )

        dataloader = datamodule.test_dataloader()
        results = trainer.test(module, dataloaders=dataloader)
        print(results)


if __name__ == "__main__":
    fire.Fire(main)

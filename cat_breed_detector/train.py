import fire
import pytorch_lightning as pl
import torch
from hydra import compose, initialize

from modules.data_module import ViTDataModule
from modules.module import ViTClassifier
from utilities.logging_selector import get_logger
from utilities.model_getter import get_model


def main():
    with initialize(config_path="../configs", version_base="1.1"):
        config = compose(config_name="config")

        pl.seed_everything(config["training"]["seed"])
        datamodule = ViTDataModule(
            config,
        )
        datamodule.prepare_data()
        datamodule.setup()

        logger = get_logger(config["logging"])

        assert len(datamodule.labels) <= config["model"]["num_labels"]
        model = get_model(datamodule)

        module = ViTClassifier(
            model,
            datamodule=datamodule,
            train_parameters=config["training"],
            log_parameters=config["logging"],
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
            callbacks=[checkpoint_callback],
        )
        trainer.fit(module, datamodule=datamodule)

        if trainer.is_global_zero:
            module.convert_to_onnx(
                output_dir=config["model"]["model_local_path"],
                model_name=config["model"]["model_onnx_name"],
            )


if __name__ == "__main__":
    fire.Fire(main)

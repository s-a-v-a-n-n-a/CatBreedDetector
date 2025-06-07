import json
from pathlib import Path

import fire
import pytorch_lightning as pl
import torch
from hydra import compose, initialize

from modules.infer_data_module import InferDataModule
from modules.module import ViTClassifier
from utilities.model_getter import get_model


def main(models_dir: Path, checkpoint_name: str, images_to_analyze: any) -> None:
    torch.set_float32_matmul_precision("medium")
    with initialize(config_path="../configs", version_base="1.1"):
        config = compose(config_name="config")
        try:
            datamodule = InferDataModule(config, images_to_analyze)
            datamodule.prepare_data()
            datamodule.setup()

            model = ViTClassifier.load_from_checkpoint(
                Path(models_dir) / checkpoint_name,
                model=get_model(datamodule),
                datamodule=datamodule,
                train_parameters=config["training"],
                log_parameters=config["logging"],
                for_test=True,
            )
            trainer = pl.Trainer(accelerator="gpu", devices="auto")

            batch_results = trainer.predict(model, datamodule=datamodule)
            print(json.dumps(batch_results, indent=4, ensure_ascii=False))
        except Exception as exception:
            print("Something went wrong:\n\t", exception)


if __name__ == "__main__":
    fire.Fire(main)

import pytorch_lightning as pl
import torchmetrics
import torch
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

from pathlib import Path
import typing as tp
import git


class ViTClassifier(pl.LightningModule):
    """Module for training, evaluation and testing models
    for the classification task.
    """

    def __init__(
            self,
            model,
            datamodule: pl.LightningDataModule,
            train_parameters: dict[str, any],
            log_parameters: dict[str, any],
            for_test: bool = False
    ) -> None:
        super().__init__()
        self.model = model

        self.lr = train_parameters["lr"]
        self.weight_decay = train_parameters["weight_decay"]
        self.num_labels = len(datamodule.id2label)

        self._log_parameters = log_parameters
        self._log_parameters = {
            "all_logs": self._log_parameters["mlflow_save_dir"]\
            if self._log_parameters["label"] == "mlflow"\
            else self._log_parameters["save_dir"],
            **self._log_parameters
        }
        self.criterion = torch.nn.CrossEntropyLoss()
        if not for_test:
            self.train_losses = []
            self.train_accuracies = []
            self.train_f1scores = []
            self.train_accuracy = torchmetrics.Accuracy(
                task="multiclass",
                num_classes=self.num_labels
            )
            self.train_f1score = torchmetrics.F1Score(
                task="multiclass",
                num_classes=self.num_labels,
                average='weighted'
            )
            self.val_losses = []
            self.val_accuracies = []
            self.val_f1scores = []
            self.val_accuracy = torchmetrics.Accuracy(
                task="multiclass",
                num_classes=self.num_labels
            )
            self.val_f1score = torchmetrics.F1Score(
                task="multiclass",
                num_classes=self.num_labels,
                average='weighted'
            )
            self.num_epochs = train_parameters["num_epochs"]
            self.num_training_steps = self.num_epochs * len(
                datamodule.train_dataloader()
            )
            self.num_warmup_steps = int(
                train_parameters["num_warmup_steps_ratio"] * self.num_training_steps
            )
            self.batch_size = train_parameters["batch_size"]
            self.save_custom_parameters()
        if for_test:
            self.test_accuracy = torchmetrics.Accuracy(
                task="multiclass",
                num_classes=self.num_labels
            )
            self.test_f1score = torchmetrics.F1Score(
                task="multiclass",
                num_classes=self.num_labels,
                average='weighted'
            )
        self.datamodule = datamodule

    def save_custom_parameters(self):
        config = self.model.config
        hyperparams = {
            k: v for k, v in config.to_dict().items() 
            if not k.startswith("_") and not callable(v)
        }

        training_hyperparams = {
            "learning_rate": self.lr,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
        }
        hyperparams.update(training_hyperparams)
        hyperparams.update(self.log_version())
        self.save_hyperparameters(hyperparams)

        # params = training_hyperparams
        # params = {**params, **self.log_version()}
        # self.save_parameters(params)
        # logger = self.logger.experiment
        # logger.log_params(hyperparams)
        # self.log_version(logger)

    def log_version(self) -> dict:
        repo = git.Repo(search_parent_directories=True)
        return {"git_commit_id": repo.head.object.hexsha}

    # def log_model(self):
    #     logger = self.logger.experiment
    #     logger.pytorch.log_model(self.model, "model")

    def forward(self, x):
        return self.model(pixel_values=x).logits

    def visualize(
            self,
            title: str,
            label: str,
            losses: list[float],
            to_save: Path
    ) -> None:
        plt.figure()
        plt.plot(losses, label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(title)
        plt.legend()
        plt.savefig(to_save)
        plt.close()
        # mlflow.load_artifact(to_save)

    def common_step(
            self,
            batch,
            batch_idx
    ) -> tuple:
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        preds = self(pixel_values)
        loss = self.criterion(preds, labels)
        preds = preds.argmax(-1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.common_step(batch, batch_idx)

        self.train_losses.append(loss.detach().cpu().numpy())

        self.train_accuracy.update(preds, labels)
        self.train_f1score.update(preds, labels)

        self.train_accuracies.append(self.train_accuracy.compute().item())
        self.train_f1scores.append(self.train_f1score.compute().item())

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )
        self.log(
            "train_accuracy",
            self.train_accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )
        self.log(
            "train_f1score",
            self.train_f1score,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.common_step(batch, batch_idx)
        self.val_f1score.update(preds, labels)
        self.val_accuracy.update(preds, labels)

        self.val_losses.append(loss.detach().cpu().numpy())
        self.val_accuracies.append(self.val_accuracy.compute().item())
        self.val_f1scores.append(self.val_f1score.compute().item())

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )
        self.log(
            "val_accuracy",
            self.val_accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )
        self.log(
            "val_f1score",
            self.val_f1score,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )
        return {
            "val_loss": loss,
            "val_accuracy": self.val_accuracy,
            "val_f1score": self.val_f1score
        }

    def on_train_end(self):
        plots_path = Path(
            self._log_parameters["all_logs"]
        ) / self._log_parameters["plots_path"]
        Path(plots_path).mkdir(parents=True, exist_ok=True)
        self.visualize(
            "Train loss",
            "Train loss",
            self.train_losses,
            Path(plots_path) / "train_loss.png"
        )
        self.visualize(
            "Train accuracy",
            "Train accuracy",
            self.train_accuracies,
            Path(plots_path) / "train_accuracy.png"
        )
        self.visualize(
            "Train f1score",
            "Train f1score",
            self.train_f1scores,
            Path(plots_path) / "train_f1score.png"
        )
        # self.log_model()

    def on_validation_end(self):
        plots_path = Path(
            self._log_parameters["all_logs"]
        ) / self._log_parameters["plots_path"]
        Path(plots_path).mkdir(parents=True, exist_ok=True)
        self.visualize(
            "Val loss",
            "Val loss",
            self.val_losses,
            Path(plots_path) / "val_loss.png"
        )
        self.visualize(
            "Val accuracy",
            "Val accuracy",
            self.val_accuracies,
            Path(plots_path) / "val_accuracy.png"
        )
        self.visualize(
            "Val f1score",
            "Val f1score",
            self.val_f1scores,
            Path(plots_path) / "val_f1score.png"
        )

    def test_step(self, batch, batch_idx):
        _, preds, labels = self.common_step(batch, batch_idx)

        self.test_accuracy.update(preds, labels)
        self.test_f1score.update(preds, labels)
        self.log(
            "test_accuracy",
            self.test_accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )
        self.log(
            "test_f1score",
            self.test_f1score,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )
        return {
            "test_accuracy": self.test_accuracy,
            "test_f1score": self.test_f1score
        }

    def predict_step(
            self,
            batch: tp.Any,
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> tp.Any:
        pixel_values = batch["pixel_values"]
        preds = self(pixel_values)
        probabilities = torch.nn.functional.softmax(preds, dim=1)
        prediction_class = torch.argmax(probabilities).item()
        return {
            # "class_id": pred_class,
            "class_name": self.datamodule.id2label[str(int(prediction_class))],
            "confidence": probabilities[0][prediction_class].item(),
            # "probabilities": {
            #     cls_name: probs[0][i].item() 
            #     for i, cls_name in enumerate(self.class_names)
            # }
        }

    def get_optimizer(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.val_dataloader()

    def predict_dataloader(self):
        return self.datamodule.predict_dataloader()

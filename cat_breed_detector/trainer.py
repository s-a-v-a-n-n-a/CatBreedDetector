import pytorch_lightning as pl
import torchmetrics
import torch
from transformers import get_linear_schedule_with_warmup


class CatBreedClassifier(pl.LightningModule):
    """Module for training, evaluation and testing models
    for the classification task
    """

    def __init__(
            self,
            model,
            num_labels: int,
            lr: float,
            weight_decay: float,
            num_warmup_steps: int,
            num_training_steps: int,
            datamodule: pl.LightningDataModule,
            for_test: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = torch.nn.CrossEntropyLoss()
        if not for_test:
            self.train_accuracy = torchmetrics.Accuracy(
                task="multiclass",
                num_classes=num_labels
            )
            self.val_accuracy = torchmetrics.Accuracy(
                task="multiclass",
                num_classes=num_labels
            )
            self.train_f1score = torchmetrics.F1Score(
                task="multiclass",
                num_classes=num_labels,
                average='weighted'
            )
            self.val_f1score = torchmetrics.F1Score(
                task="multiclass",
                num_classes=num_labels,
                average='weighted'
            )
        if for_test:
            self.test_accuracy = torchmetrics.Accuracy(
                task="multiclass",
                num_classes=num_labels
            )
            self.test_f1score = torchmetrics.F1Score(
                task="multiclass",
                num_classes=num_labels,
                average='weighted'
            )
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.datamodule = datamodule

    def forward(self, x):
        return self.model(pixel_values=x).logits

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        preds = self(pixel_values)
        loss = self.criterion(preds, labels)
        preds = preds.argmax(-1)
        self.train_accuracy.update(preds, labels)
        self.train_f1score.update(preds, labels)

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
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        preds = self(pixel_values)
        loss = self.criterion(preds, labels)

        preds = preds.argmax(-1)
        self.val_f1score.update(preds, labels)
        self.val_accuracy.update(preds, labels)
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

    def test_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        preds = self(pixel_values)
        loss = self.criterion(preds, labels)

        preds = preds.argmax(1)
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

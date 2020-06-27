# pylint: disable=invalid-name,missing-docstring,missing-class-docstring,arguments-differ,C0330,too-many-locals,too-many-ancestors,too-many-instance-attributes
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl

_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(_FILE_PATH, "..", "data")


class Classifier(pl.LightningModule):
    def __init__(self, hparams):
        super(Classifier, self).__init__()
        self.hparams = hparams

        self.main = nn.Sequential(
            # batch x (nc) x IMAGE_SIZE x IMAGE_SIZE
            nn.Conv2d(hparams.nc, hparams.nf, 4, 2, 1),
            # batch x (nf) x IMAGE_SIZE//2 x IMAGE_SIZE//2
            nn.ELU(),
            nn.MaxPool2d(2),
            # batch x (nf) x IMAGE_SIZE//4 x IMAGE_SIZE//4
            nn.Conv2d(hparams.nf, hparams.nf * 2, 4, 2, 1),
            # batch x (nf * 4) x IMAGE_SIZE//8 x IMAGE_SIZE//8
            nn.ELU(),
            nn.MaxPool2d(2),
            # batch x (nf * 4) x IMAGE_SIZE//16 x IMAGE_SIZE//16
            nn.Dropout2d(p=0.5),
            nn.Flatten(),
            nn.Linear(
                hparams.nf
                * 2
                * (hparams.image_size // 16)
                * (hparams.image_size // 16),
                10,
            ),
            nn.LogSoftmax(dim=1),
        )
        self.criterion = F.nll_loss

    def forward(self, x):
        return self.main(x)

    def training_step(self, batch, _batch_idx):
        x, y_ref = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y_ref)
        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, _batch_idx):
        x, y_ref = batch
        y_pred = self(x)
        val_loss = self.criterion(y_pred, y_ref)
        pred = y_pred.argmax(dim=1, keepdim=True)
        total_correct = pred.eq(y_ref.view_as(pred)).sum().item()
        return {
            "val_loss": val_loss,
            "total_correct": total_correct,
            "total": y_ref.size(0),
        }

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        total_correct = sum(x["total_correct"] for x in outputs)
        total = sum(x["total"] for x in outputs)
        val_accuracy = total_correct / total
        log = {"val_loss": val_loss, "val_accuracy": val_accuracy}
        return {"log": log, "val_loss": val_loss}

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        """Implicit PyTorch Lightning train dataloader definition"""
        train_set = datasets.MNIST(
            self.hparams.data,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(self.hparams.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5),
                ]
            ),
        )
        return DataLoader(
            train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.workers,
        )

    def val_dataloader(self):
        """Implicit PyTorch Lightning valid dataloader definition"""
        valid_set = datasets.MNIST(
            self.hparams.data,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(self.hparams.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5),
                ]
            ),
        )
        return DataLoader(
            valid_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.workers,
        )


def arg_parser():
    parser = argparse.ArgumentParser(description="Trains an experimental model...")
    parser.add_argument("-lr", default=0.0002, type=float, help="learning rate")
    parser.add_argument("--batch-size", default=256, type=int, help="mini batch size")
    parser.add_argument("--seed", default=42, type=float, help="seed for randomness")
    parser.add_argument("--workers", default=4, type=int, help="workers for dataloader")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, type=str)
    parser.add_argument("--nc", default=1, type=int, help="channels in input data")
    parser.add_argument("--nf", default=64, type=int, help="features for conv model")
    parser.add_argument("--image-size", default=28, type=int, help="image resizing")

    return parser


def main():
    parser = arg_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    classifier = Classifier(args)
    trainer = pl.Trainer.from_argparse_args(
        args,
        row_log_interval=1,
        early_stop_callback=pl.callbacks.EarlyStopping(patience=3),
    )
    trainer.fit(classifier)


if __name__ == "__main__":
    main()

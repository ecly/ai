# pylint: disable=invalid-name,missing-docstring,missing-class-docstring,arguments-differ,C0330,too-many-locals
"""Partially based on: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import pytorch_lightning as pl


_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(_FILE_PATH, "..", "data")
IMAGE_SIZE = 32


class Generator(nn.Module):
    """Minimalistic Generator implementation"""

    def __init__(self, hparams):
        super(Generator, self).__init__()
        self.hparams = hparams

        self.main = nn.Sequential(
            # input is latent vector Z, going into a convolution
            nn.ConvTranspose2d(hparams.nz, hparams.ngf * 4, 4, 1, 0),
            nn.BatchNorm2d(hparams.ngf * 4),
            nn.ELU(),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(hparams.ngf * 4, hparams.ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(hparams.ngf * 2),
            nn.ELU(),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(hparams.ngf * 2, hparams.ngf, 4, 2, 1),
            nn.BatchNorm2d(hparams.ngf),
            nn.ELU(),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(hparams.ngf, hparams.nc, 4, 2, 1),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """Minimalistic Discriminator implementation"""

    def __init__(self, hparams):
        super(Discriminator, self).__init__()
        self.hparams = hparams

        self.main = nn.Sequential(
            # input size. (nc) x 32 x 32
            nn.Conv2d(hparams.nc, hparams.ndf, 4, 2, 1),
            nn.BatchNorm2d(hparams.ndf),
            nn.ELU(),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(hparams.ndf, hparams.ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(hparams.ndf * 2),
            nn.ELU(),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(hparams.ndf * 2, hparams.ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(hparams.ndf * 4),
            nn.ELU(),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(hparams.ndf * 4, 1, 4, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)


class DCGAN(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """A PyTorch Lightning wrapper for training G and D"""

    def __init__(self, args):
        super(DCGAN, self).__init__()
        self.hparams = args
        self.g = Generator(args)
        self.d = Discriminator(args)
        # use for generating samples showing progress over time
        self.fixed_z = torch.randn(64, args.nz, 1, 1)
        self.criterion = F.binary_cross_entropy

    def forward(self, x):
        return self.g(x)

    def _generator_step(self, batch):
        """Definition of a single training step for G"""
        b_size = batch[0].size(0)
        device = torch.device("cuda" if self.on_gpu else "cpu")

        # If optimizer_idx != 0 we are training G
        z = torch.randn(b_size, self.hparams.nz, 1, 1, device=device)
        fake_x = self.g(z)
        gen_output = self.d(fake_x).view(-1)
        gen_label = torch.ones((b_size,), device=device)
        err_g = self.criterion(gen_output, gen_label)
        log_dict = {"g_loss": err_g}

        return {"loss": err_g, "progress_bar": log_dict, "log": log_dict}

    def _discriminator_step(self, batch):
        """Definition of a single training step for D"""
        x, _ = batch
        b_size = x.size(0)
        device = torch.device("cuda" if self.on_gpu else "cpu")

        ## Forward pass real batch through D
        real_output = self.d(x).view(-1)
        real_label = torch.ones((b_size,), device=device)
        real_err = self.criterion(real_output, real_label)

        ## Generate fake batch and forward pass through D
        z = torch.randn(b_size, self.hparams.nz, 1, 1, device=device)
        fake_x = self.g(z)
        fake_label = torch.zeros((b_size,), device=device)
        fake_output = self.d(fake_x.detach()).view(-1)
        fake_err = self.criterion(fake_output, fake_label)

        err_d = real_err + fake_err
        log_dict = {"d_loss": err_d}
        return {"loss": err_d, "progress_bar": log_dict, "log": log_dict}

    def training_step(self, batch, _batch_idx, optimizer_idx):
        """Training step definition for multi-optimizer PyTorch Lightning"""
        if optimizer_idx == 0:
            return self._discriminator_step(batch)

        return self._generator_step(batch)

    def configure_optimizers(self):
        """For GANs we return two optimizer, order coupled with train step"""
        optimizer_d = Adam(self.d.parameters(), lr=self.hparams.lr)
        optimizer_g = Adam(self.g.parameters(), lr=self.hparams.lr)
        return [optimizer_d, optimizer_g], []

    def train_dataloader(self):
        """Implicit PyTorch Lightning train dataloader definition"""
        train_set = datasets.MNIST(
            self.hparams.data,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(IMAGE_SIZE),
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

    def on_epoch_end(self):
        """
        After every epoch we generate and log our images on fixed noise
        to see progress on the same latent vector over time.
        """
        z = self.fixed_z
        if self.on_gpu:
            z = z.to(torch.device("cuda"))

        fake_x = self.g(z)
        grid = vutils.make_grid(fake_x)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4, help="workers for dataloader")
    parser.add_argument("--batch-size", type=int, default=256, help="mini batch size")
    parser.add_argument("--nc", type=int, default=1, help="number of channels in image")
    parser.add_argument("--nz", type=int, default=100, help="size of z (latent vector)")
    parser.add_argument("--ngf", type=int, default=32, help="size of feature map in g")
    parser.add_argument("--ndf", type=int, default=32, help="size of feature map in d")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    return parser


def main():
    parser = arg_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    gan = DCGAN(args)
    trainer = pl.Trainer.from_argparse_args(args, row_log_interval=1)
    trainer.fit(gan)


if __name__ == "__main__":
    main()

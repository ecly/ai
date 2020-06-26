# pylint: disable=invalid-name,missing-docstring,missing-class-docstring,arguments-differ,too-many-instance-attributes,too-many-locals
"""
Partially based on:
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
import argparse
import random
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_LABEL = 1
FAKE_LABEL = 0


def manual_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        nz = args.nz  # latent vector size
        ngf = args.ngf
        nc = args.nc

        self.main = nn.Sequential(
            # input is latent vector Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        ndf = args.ndf
        nc = args.nc

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)


def get_mnist(path, image_size=64, batch_size=16, workers=4):
    train_set = datasets.MNIST(
        path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    valid_set = datasets.MNIST(
        path,
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    return train_loader, valid_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers", type=int, default=4, help="workers for data loader"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="the h/w of the input image to network",
    )
    parser.add_argument("--nc", type=int, default=1, help="number of channels in image")
    parser.add_argument("--nz", type=int, default=100, help="size of z (latent vector)")
    parser.add_argument(
        "--ngf", type=int, default=64, help="size of feature map in generator"
    )
    parser.add_argument(
        "--ndf", type=int, default=64, help="size of feature map in generator"
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for Adam")
    return parser.parse_args()


def visualize_images(images, path):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(
        np.transpose(
            vutils.make_grid(images.to(DEVICE)[:64], padding=2, normalize=True).cpu(),
            (1, 2, 0),
        )
    )
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_losses(d_losses, g_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.legend()
    plt.show()
    plt.waitforbuttonpress()


def main():
    args = parse_args()
    manual_seed(args.seed)

    train_loader, _valid_loader = get_mnist(
        args.data, image_size=args.image_size, batch_size=args.batch_size
    )

    net_g = Generator(args).to(DEVICE)
    net_d = Discriminator(args).to(DEVICE)
    criterion = nn.BCELoss()

    # Batch of latent vectors used for visualizing progress
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=DEVICE)

    optimizer_d = Adam(net_d.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_g = Adam(net_g.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    g_losses = []
    d_losses = []

    for i, (real_x, _y_ref) in enumerate(cycle(train_loader)):
        if i > args.iterations:
            break

        real_x = real_x.to(DEVICE)
        b_size = real_x.size(0)

        # Train discriminator for one batch
        net_d.zero_grad()

        ## Forward pass real batch through D
        real_output = net_d(real_x).view(-1)
        real_label = torch.full(
            (real_x.size(0),), REAL_LABEL, dtype=torch.float32, device=DEVICE
        )
        real_err = criterion(real_output, real_label)

        ## Forward pass fake batch through D
        noise = torch.randn(b_size, args.nz, 1, 1, device=DEVICE)
        fake_x = net_g(noise)
        fake_label = torch.full(
            (fake_x.size(0),), FAKE_LABEL, dtype=torch.float32, device=DEVICE
        )
        fake_output = net_d(fake_x.detach()).view(-1)
        fake_err = criterion(fake_output, fake_label)

        err_d = real_err + fake_err
        err_d.backward()
        optimizer_d.step()

        ## Forward pass through G using D as criterion
        net_g.zero_grad()
        output = net_d(fake_x).view(-1)
        labels = torch.full((b_size,), REAL_LABEL, dtype=torch.float32, device=DEVICE)
        err_g = criterion(output, labels)
        err_g.backward()
        optimizer_g.step()

        # Output training stats
        if i % 10 == 0:
            print(
                f"[{i}/{args.iterations}] loss_d: {err_d.item():.3f}, loss_g: {err_g.item():.3f}"
            )

        # Test generator performance on fixed_noise
        if i % 50 == 0:
            with torch.no_grad():
                fake = net_g(fixed_noise).detach().cpu()
                path = f"gen_img_{i}.png"
                visualize_images(fake, path)

        # Save Losses for plotting later
        g_losses.append(err_d.item())
        d_losses.append(err_g.item())

    plot_losses(d_losses, g_losses)
    torch.save(net_g.state_dict(), f"generator_iter_{args.iterations}.pt")
    torch.save(net_d.state_dict(), f"discriminator_iter_{args.iterations}.pt")


if __name__ == "__main__":
    main()

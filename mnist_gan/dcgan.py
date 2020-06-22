# pylint: disable=invalid-name,missing-docstring,missing-class-docstring,arguments-differ,too-many-instance-attributes,too-many-locals
"""
Partially based on:
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import torch

# import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# use interactive matplotlib mode
plt.ion()


def manual_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.nz = args.nz
        ngf = args.ngf
        self.ngf = ngf
        nc = args.nc

        # we use 7 to have easy scaling to 28 x 28 for mnist
        self.proj0 = nn.Linear(self.nz, 7 * 7 * ngf * nc)
        self.bn0 = nn.BatchNorm1d(7 * 7 * ngf * nc)
        self.act0 = nn.ELU()

        self.conv1 = nn.ConvTranspose2d(
            in_channels=ngf * nc,
            out_channels=ngf * 2,
            kernel_size=6,
            stride=2,
            padding=2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(ngf * 2)
        self.act1 = nn.ELU()

        self.conv2 = nn.ConvTranspose2d(
            in_channels=ngf * 2,
            out_channels=1,
            kernel_size=6,
            stride=2,
            padding=2,
            bias=False,
        )
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.proj0(x.squeeze(3).squeeze(2))
        x = self.bn0(x)
        x = self.act0(x)
        x = x.view(x.shape[0], self.ngf, 7, 7)
        # batch x ngf x 7 x 7

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        # batch x ngf * 2 x 14 x 14

        x = self.conv2(x)
        x = self.act2(x)

        # batch x 1 x 28 x 28
        return x


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        ndf = args.ndf
        nc = args.nc

        self.conv1 = nn.Conv2d(nc, ndf, 5, 2, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf)
        self.act1 = nn.ELU()

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.act2 = nn.ELU()

        self.conv3 = nn.Conv2d(ndf * 2, 1, 7, 1, 0, bias=False)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        # batch x nc x 28 x 28
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        # batch x ndf x 14 x 14
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        # batch x ndf * 2 x 7
        x = self.conv3(x)
        x = self.act3(x)

        # batch x 1 x 1 x 1
        return x


def get_mnist(path, batch_size=16, workers=4):
    train_set = datasets.MNIST(
        path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    valid_set = datasets.MNIST(
        path,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
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


def visualize_images(images, path=None):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(
        np.transpose(
            vutils.make_grid(images.to(DEVICE)[:64], padding=2, normalize=True).cpu(),
            (1, 2, 0),
        )
    )
    if path is not None:
        plt.savefig(path, bbox_inches="tight")
    else:
        plt.waitforbuttonpress()

def plot_losses(d_losses, g_losses):

def main():
    args = parse_args()
    manual_seed(args.seed)

    train_loader, _valid_loader = get_mnist(args.data, batch_size=args.batch_size)

    net_g = Generator(args).to(DEVICE)
    net_d = Discriminator(args).to(DEVICE)
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=DEVICE)

    # Establish convention for real and fake labels during training
    REAL_LABEL = 1
    FAKE_LABEL = 0

    # Setup Adam optimizers for both G and D
    optimizer_d = optim.Adam(net_d.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_g = optim.Adam(net_g.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    g_losses = []
    d_losses = []

    iters = 0
    while iters < args.iterations:
        for real_x, _y_ref in train_loader:
            real_x = real_x.to(DEVICE)
            b_size = real_x.size(0)

            # Train discriminator for one batch
            net_d.zero_grad()

            ## Forward pass real batch through D
            real_output = net_d(real_x).view(-1)
            real_label = torch.full((real_x.size(0),), REAL_LABEL, device=DEVICE)
            real_err = criterion(real_output, real_label)

            ## Forward pass fake batch through D
            noise = torch.randn(b_size, args.nz, 1, 1, device=DEVICE)
            fake_x = net_g(noise)
            fake_label = torch.full((fake_x.size(0),), FAKE_LABEL, device=DEVICE)
            fake_output = net_d(fake_x.detach()).view(-1)
            fake_err = criterion(fake_output, fake_label)

            err_d = real_err + fake_err
            err_d.backward()
            optimizer_d.step()

            # Train generator for one batch
            net_g.zero_grad()
            output = net_d(fake_x).view(-1)
            labels = torch.full((b_size,), REAL_LABEL, device=DEVICE)
            err_g = criterion(output, labels)
            err_g.backward()
            optimizer_g.step()

            # Output training stats
            if iters % 10 == 0:
                print(
                    f"[{iters}/{args.iterations}] loss_d: {err_d.item():.3f}, loss_g: {err_g.item():.3f}"
                )

            iters += 1

            # Save Losses for plotting later
            g_losses.append(err_d.item())
            d_losses.append(err_g.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if iters % 50 == 0:
                with torch.no_grad():
                    fake = net_g(fixed_noise).detach().cpu()
                    path = f"gen_img_{iters}.png"
                    visualize_images(fake, path)

    # plt.figure(figsize=(10, 5))
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(g_losses, label="G")
    # plt.plot(d_losses, label="D")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()

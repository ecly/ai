import time
import argparse
from statistics import mean
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.utils.data.distributed
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pylint: disable=C0330


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216, 128)

        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def validate(model, criterion, valid_loader):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_valid = len(valid_loader.dataset)
    with torch.no_grad():
        for x, y_ref in tqdm(valid_loader, desc="Validating", leave=False):
            x, y_ref = x.to(DEVICE), y_ref.to(DEVICE)
            y_pred = model(x.to(DEVICE))
            total_loss = criterion(y_pred, y_ref, reduction="sum").item()
            pred = y_pred.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(y_ref.view_as(pred)).sum().item()

    avg_loss = total_loss / total_valid
    print(
        "Avg. validation loss {:.2f}, Correct predictions {}/{}:".format(
            avg_loss, total_correct, total_valid
        )
    )


def train(
    model,
    optimizer,
    train_loader,
    valid_loader,
    epochs=10,
    log_interval=100,
    validate_every=1000,
):
    """Start training"""
    # run one epoch
    batch_size = train_loader.batch_size
    dataset_size = len(train_loader.dataset)
    start = time.time()
    for epoch in range(epochs):
        losses = []
        for idx, (x, y_ref) in enumerate(train_loader):
            x, y_ref = x.to(DEVICE), y_ref.to(DEVICE)

            # make sure gradients are zeroed before model is applied to x
            optimizer.zero_grad()

            y_pred = model(x)
            loss = F.nll_loss(y_pred, y_ref)
            loss.mean().backward()
            losses.append(loss.item())
            optimizer.step()

            if idx % log_interval == 0:
                elapsed = int(time.time() - start)
                print(
                        "Epoch: {} [{:<5}/{:<5}] Elapsed {:03d}s Loss: {:.2f} Avg. Loss: {:.2f}".format(
                        epoch, idx * batch_size, dataset_size, elapsed, loss.item(), mean(losses)
                    )
                )

            if idx > 0 and idx % validate_every == 0:
                validate(model, F.nll_loss, valid_loader)

        validate(model, F.nll_loss, valid_loader)


def get_mnist(batch_size, workers):
    train_set = datasets.MNIST(
        "mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers
    )

    valid_set = datasets.MNIST(
        "mnist",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    return train_loader, valid_loader


def prepare_arg_parser():
    """Create arg parser handling input/output and training conditions"""
    parser = argparse.ArgumentParser(description="Trains an experimental model...")
    parser.add_argument(
        "-lr", "--learning-rate", default=0.01, type=float, help="learning rate"
    )
    parser.add_argument("--batch-size", default=160, type=int, help="mini batch size")
    parser.add_argument("--seed", default=42, type=float, help="seed for randomness")
    parser.add_argument(
        "--workers", default=8, type=int, help="number of data loading workers",
    )

    return parser


def main():
    """Build dataset according to args and train model"""
    arg_parser = prepare_arg_parser()
    args = arg_parser.parse_args()

    torch.manual_seed(args.seed)
    train_loader, valid_loader = get_mnist(args.batch_size, args.workers)

    model = Model()
    # model = torch.nn.parallel.DataParallel(model)
    model = model.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    train(model, optimizer, train_loader, valid_loader)


if __name__ == "__main__":
    main()

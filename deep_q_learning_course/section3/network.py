from torch import nn
from torch.optim import Adam


class DeepQNetwork(nn.Module):
    def __init__(self, state_dims, n_classes, hidden_dim=128, lr=0.001):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(*state_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.layers(x)

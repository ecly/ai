import random

import torch

from .network import DeepQNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        state_dims,
        n_actions,
        lr=0.0001,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.00001,
        epsilon_min=0.01,
    ):
        self.q = DeepQNetwork(state_dims, n_actions, lr=lr).to(DEVICE)
        self.n_actions = n_actions
        self.state_dims = state_dims
        self.lr = lr  # alpha
        self.discount_factor = discount_factor  # gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def predict(self, state):
        state = torch.tensor(state, device=DEVICE)
        actions = self.q(state)
        return actions

    def update(self, state, action, reward, state_):
        reward = torch.tensor(reward, device=DEVICE)

        q_next = self.predict(state_).max()

        q_hyp = self.predict(state)[action]
        q_ref = reward + self.discount_factor * q_next
        loss = self.q.criterion(q_ref, q_hyp)

        loss.backward()
        self.q.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def sample(self, state):
        if random.random() < self.epsilon:
            action = random.choice(range(self.n_actions))
        else:
            actions = self.predict(state)
            action = actions.argmax().item()

        return action

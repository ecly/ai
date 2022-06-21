import random
import numpy as np


class Agent:
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        lr=0.001,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.9999995,
        epsilon_min=0.01,
        n_actions=4,
        n_states=16,
    ):
        self.q = {s: [0.0] * n_actions for s in range(n_states)}
        self.lr = lr  # alpha
        self.discount_factor = discount_factor  # gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def update(self, state, action, reward, state_):
        self.q[state][action] += self.lr * (
            reward
            + self.discount_factor * max(self.q[state_])
            - self.q[state][action]
        )
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def sample(self, state):
        actions = self.q[state]
        if random.random() < self.epsilon:
            action = random.choice(range(len(actions)))
        else:
            action = np.argmax(actions)

        return action

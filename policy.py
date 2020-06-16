# pylint: disable=missing-docstring,missing-module-docstring,not-callable
from itertools import count
import argparse
import gym
from gym import spaces
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, output_size)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


def parse_size(x):
    if isinstance(x, spaces.Box):
        return x.shape[0]

    if isinstance(x, int):
        return x

    return x.n


def select_action(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode(policy, optimizer, gamma, eps):
    R = 0
    policy_loss = []
    returns = []
    for reward in policy.rewards[::-1]:
        R = reward + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def run(args):
    # env = gym.make("Pendulum-v0")
    env = gym.make("CartPole-v1")
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = Policy(parse_size(env.observation_space), parse_size(env.action_space))
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    eps = np.finfo(np.float32).eps.item()

    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(policy, state)
            state, reward, done, _ = env.step(action)
            if args.render and running_reward > 100:
                env.render()

            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode(policy, optimizer, args.gamma, eps)
        if i_episode % args.log_interval == 0:
            print(
                "Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}".format(
                    i_episode, ep_reward, running_reward
                )
            )
        if running_reward > env.spec.reward_threshold:
            print(
                "Solved! Running reward is now {} and "
                "the last episode runs to {} time steps!".format(running_reward, t)
            )
            break


def main():
    parser = argparse.ArgumentParser(description="PyTorch REINFORCE example")
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--seed", type=int, default=543, metavar="N", help="random seed (default: 543)"
    )
    parser.add_argument("--render", action="store_true", help="render the environment")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="interval between training status logs (default: 10)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=100,
        metavar="N",
        help="do not visualize the first N episodes (default: 100)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from common.wrappers import RepeatActionAndMaxFrameWrapper
from common.utils import plot_learning

def main():
    env = gym.make("CartPole-v1")
    env = RepeatActionAndMaxFrameWrapper(env)
    n_states = env.observation_space.shape
    n_actions = env.action_space.n
    running_win_pct = []
    scores = []
    eps = []
    agent = Agent(n_states, n_actions)
    n_games = 10_000
    for i in range(n_games):
        state = env.reset()
        score = 0
        while True:
            action = agent.sample(state)
            new_state, reward, done, _info = env.step(action)
            agent.update(state, action, reward, new_state)
            score += reward

            state = new_state

            if done:
                break

        scores.append(score)
        eps.append(agent.epsilon)
        if i % 10 == 0:
            win_pct = np.mean(scores[-100:])
            running_win_pct.append(win_pct)
            print(f"{i}: {win_pct}, {agent.epsilon}")

    filename = "cartpole_naive_dqn.png"
    plot_learning(scores, eps, filename)
    plt.show()
    env.close()


if __name__ == "__main__":
    main()

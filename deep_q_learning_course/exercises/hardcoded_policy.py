import gym
import matplotlib.pyplot as plt
import numpy as np


def main():
    env = gym.make("FrozenLake-v1")
    running_win_perc = []
    scores = []
    policy = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1, 8: 2, 9: 1, 10: 1, 13: 2, 14: 2}
    for i in range(1000):
        observation, _info = env.reset(return_info=True)
        score = 0
        while True:
            action = policy[observation]
            observation, reward, done, info = env.step(action)
            score += reward

            if done:
                break

        scores.append(score)
        if i % 10 == 0:
            running_win_perc.append(np.mean(scores[-10:]))

    plt.plot(running_win_perc)
    plt.show()
    env.close()


if __name__ == "__main__":
    main()

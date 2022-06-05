import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("WebAgg")

env = gym.make("FrozenLake-v1")
running_win_perc = []
scores = []
for i in range(1000):
    score = 0
    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        score += reward

        if done:
            break

    observation, info = env.reset(return_info=True)
    scores.append(score)
    if i % 10 == 0:
        running_win_perc.append(np.mean(scores[-10:]))

plt.plot(running_win_perc)
plt.show()
env.close()

import gym
import matplotlib.pyplot as plt
import numpy as np
from agents import q_table

def main():
    env = gym.make("FrozenLake-v1", is_slippery=True)
    running_win_pct = []
    scores = []
    agent = q_table.Agent()
    for i in range(500_000):
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
        if i % 10_000 == 0:
            win_pct = np.mean(scores[-100:])
            running_win_pct.append(win_pct)
            print(f"{i}: {win_pct}, {agent.epsilon}")

    plt.plot(running_win_pct)
    plt.show()
    env.close()


if __name__ == "__main__":
    main()

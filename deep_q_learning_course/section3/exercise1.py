import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .agent import Agent

matplotlib.use("WebAgg")


def plot_learning(scores, epsilons, filename):
    assert len(scores) == len(epsilons)
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis="x", colors="C0")
    ax.tick_params(axis="y", colors="C0")

    running_avg = []
    for t in range(len(scores)):
        running_avg.append(np.mean(scores[max(0, t - 20) : (t + 1)]))

    ax2.plot(running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Score", color="C1")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis="y", colors="C1")
    plt.savefig(filename)


def main():
    env = gym.make("CartPole-v1")
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

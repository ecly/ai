import matplotlib.pyplot as plt
import numpy as np


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

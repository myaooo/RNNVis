"""
Functions that use matplotlib to plot state stats
"""

import numpy as np
import matplotlib.pyplot as plt

from rnnvis.state_processor import compute_stats
from rnnvis.vendor import tsne


color_scheme = ['#587EB6', '#C95A5F', '#63B075', 'r', 'm', 'y']

color_scheme2 = [[201 / 255, 90 / 255, 95 / 255], [88 / 255, 126 / 255, 182 / 255],
                 [114 / 255, 189 / 255, 210 / 255], [99 / 255, 176 / 255, 117 / 255]]


def plot_words_states(states, percent=50, labels=None, rank=0, y_extents=None, save_path=None):
    """
    Plot mean and filled range of a list of states
    :param states: a list, in which each element is
        a list containing the states recordings [layer_num, layer_size] of a given word
    :param percent: the percent of the range, e.g., if 50, than we select 25% ~ 75% as the filled range
    :param labels: a list of labels, if not None, will add legend to the plot using the labels
    :param rank: select a state_list to rank, default to be the first word
    :param y_extents: a list of extent of [y_min, y_max] pairs
    :param save_path: the path to save the plot, default to be None
    :return:
    """

    if len(states) > 6:
        raise ValueError("Too many word-state pairs to plot!")
    means, _, error_l, error_u, idx = compute_stats(states[rank], True)
    layer_num = len(means)
    if labels is None:
        labels = [str(i) for i in range(len(states))]

    fig, axes = plt.subplots(nrows=layer_num, sharex=True, figsize=(15, 9))
    for k, state_list in enumerate(states):
        means, _, error_l, error_u, _ = compute_stats(state_list, False, percent)
        dim = slice(0, len(means[0]), 1)
        for j in range(layer_num):
            mean = means[j][idx[j]]
            low = mean-error_l[j][idx[j]]
            high = mean+error_u[j][idx[j]]
            axes[j].plot(range(len(mean)), mean, color_scheme[k], linewidth=1, alpha=0.8, label=labels[k])
            axes[j].plot(range(len(mean)), low, color_scheme[k], linewidth=1, alpha=0.3)
            axes[j].plot(range(len(mean)), high, color_scheme[k], linewidth=1, alpha=0.3)
            axes[j].fill_between(range(dim.start, dim.stop, dim.step), low[dim], high[dim],
                                 facecolor=color_scheme[k], alpha=0.2)

    for j in range(layer_num):
        axes[j].plot([0, len(means[0])], [0, 0], 'k', linewidth=1)
        axes[j].legend()
        if y_extents is not None:
            axes[j].set_ylim(y_extents[j])
    if save_path is None:
        plt.draw()
    else:
        plt.savefig(save_path, bbox_inches='tight')


def parallel_coord(states, every=5, rank=0, y_extents=None, save_path=None):
    """
    Plot parallel coordinate of states
    :param states: a list, in which each element is
        a list containing the states recordings [layer_num, layer_size] of a given word
    :param every: plot every, skip some of the middle dims for clarity
    :param rank: select a state_list to rank, default to be the first word
    :param y_extents:  a list of extent of [y_min, y_max] pairs
    :param save_path: the path to save the plot, default to be None
    :return:
    """

    if len(states) > 6:
        raise ValueError("Too many word-state pairs to plot!")
    means, _, _, _, idx = compute_stats(states[rank], True)
    layer_num = len(means)

    fig, axes = plt.subplots(nrows=layer_num, sharex=True, figsize=(15, 9))

    num = [len(state_list) for state_list in states]  # the number of state vector in each list
    dim = slice(0, len(means[0]), every)
    h_scale = range(len(means[0]))

    for k, state_list in enumerate(states):
        for j in range(layer_num):
            for state in state_list:
                axes[j].plot(h_scale[dim], state[j][idx[j]][dim], color_scheme[k],
                             linewidth=1, alpha=(1.0 / num[k])**0.7 * 0.8)
    for j in range(layer_num):
        axes[j].plot([0, len(means[0])], [0, 0], 'k', linewidth=1)
        axes[j].legend()
        if y_extents is not None:
            axes[j].set_ylim(y_extents[j])
    if save_path is None:
        plt.draw()
    else:
        plt.savefig(save_path, bbox_inches='tight')


def scatter(states, labels=None, every=5, rank=0, y_extents=None, save_path=None):

    if len(states) > 6:
        raise ValueError("Too many word-state pairs to plot!")
    means, _, _, _, idx = compute_stats(states[rank], True)
    layer_num = len(means)
    if labels is None:
        labels = [str(i) for i in range(len(states))]

    fig, axes = plt.subplots(nrows=layer_num, sharex=True, figsize=(15, 9))

    num = [len(state_list) for state_list in states]  # the number of state vector in each list
    alphas = [(1.0 / n) ** 0.6 * 0.8 for n in num]
    dim = slice(0, len(means[0]), every)
    h_scale = list(range(len(means[0])))

    for k, state_list in enumerate(states):
        for j in range(layer_num):
            state_val = np.hstack([state[j][idx[j]][dim] for state in state_list])
            _h_scale = h_scale[dim] * num[k]
            axes[j].scatter(_h_scale, state_val, 4, color_scheme[k], alpha=alphas[k], label=labels[k])

    for j in range(layer_num):
        axes[j].plot([0, len(means[0])], [0, 0], 'k', linewidth=1)
        axes[j].legend()
        if y_extents is not None:
            axes[j].set_ylim(y_extents[j])
    if save_path is None:
        plt.draw()
    else:
        plt.savefig(save_path, bbox_inches='tight')


def create_animated_tsne(data, perplexity, states_num, init_dim=50, lr=50, max_iter=1000, path=None):

    tsne_solver = tsne.TSNE(2, perplexity, lr)
    tsne_solver.set_inputs(data, init_dim)
    projected = []
    for i in range(max_iter//10):
        cost = tsne_solver.step(10)
        projected.append(tsne_solver.get_solution())
        print("iteration: {:d}, error: {:f}".format(i*10, cost))

    color_list = []
    for i, num in enumerate(states_num):
        color_list.append(np.tile(np.array(color_scheme2[i], np.float32), (num, 1)))
    color = np.vstack(color_list)

    projected = [np.hstack((project, color)) for project in projected]
    anim = AnimatedScatter(projected, [6, 6], 50)

    if path is not None:
        anim.save(path)
    anim.show()


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, data, figsize=None, interval=5):
        from matplotlib import animation
        # self.stream = self.data_stream()
        self.figsize = [6, 6] if figsize is None else figsize
        self.data = data
        self.scat = None
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.setup_plot()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, len(data), interval=interval, repeat_delay=1000,
                                           blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        xy = self.data[0]
        self.scat = self.ax.scatter(xy[:, 0], xy[:, 1], 8, xy[:, 2:], animated=True, alpha=0.5)
        self.ax.axis([-self.figsize[0]/2.0, self.figsize[0]/2.0, -self.figsize[1]/2.0, self.figsize[1]/2.0])

        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self, i):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        if callable(self.data):
            return self.data()
        else:
            return self.data[i]

    def update(self, i):
        """Update the scatter plot."""
        data = self.data_stream(i)
        min_x = np.min(data[:, 0])
        max_x = np.max(data[:, 0])
        min_y = np.min(data[:, 1])
        max_y = np.max(data[:, 1])
        cent_x = (max_x + min_x)/2
        cent_y = (max_y + min_y)/2
        scale = 0.9 * min(self.figsize[0], self.figsize[1]) / max(max_x - min_x, max_y - min_y)
        data[:, 0] -= cent_x
        data[:, 1] -= cent_y
        data *= scale

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        return self.scat,

    def show(self):
        plt.show()

    def save(self, filename, fps=15, bitrate=1800):
        from matplotlib import animation
        _writer = animation.writers['ffmpeg']
        writer = _writer(fps=fps, metadata=dict(artist='Ming'), bitrate=bitrate)
        self.ani.save(filename, writer)
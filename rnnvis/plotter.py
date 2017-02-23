"""
Functions that use matplotlib to plot state stats
"""

import numpy as np
import matplotlib.pyplot as plt

from rnnvis.state_processor import compute_stats


color_scheme = ['#587EB6', '#C95A5F', '#63B075', 'r', 'm', 'y']


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

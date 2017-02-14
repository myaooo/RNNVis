"""
Tests the hidden state correlation
"""

import pickle

# import tensorflow as tf
import numpy as np

from rnnvis.procedures import build_model, init_tf_environ, pour_data
from rnnvis.state_processor import load_words_and_state, sort_by_id, compute_stats


def find_candidate(means, stds, k):
    means = np.vstack(means)
    stds = np.vstack(stds)
    candidate = []
    stds_ = []
    mean_ = []
    for i in range(means.shape[1]):
        idx = np.argpartition(-np.abs(means[:, i]), k)[:k]
        mean = means[idx, i]
        idx_ = np.argsort(-np.abs(mean))
        idx = idx[idx_]
        candidate.append(idx.tolist())
        mean_.append(means[idx, i].tolist())
        stds_.append(stds[idx, i].tolist())
    return candidate, mean_, stds_


def get_model_params(model_config):
    model, train_config = build_model(model_config, True)
    model.restore()
    embedding = model.embedding_weights
    return embedding


colors = ['b', 'g', 'r', 'c', 'm', 'y']


def plot_words_states(id_states, ids):

    import matplotlib.pyplot as plt

    stds, means, error_l, error_u, idx = compute_stats(id_states[ids[0]], True)
    layer_num = len(means)

    fig, axes = plt.subplots(nrows=layer_num, sharex=True, figsize=(15, 9))
    for k, id_ in enumerate(ids):
        state = id_states[id_]
        stds, means, error_l, error_u, _ = compute_stats(state, False)
        dim = slice(0, len(means[0]), 1)
        for j in range(layer_num):
            mean = means[j][idx[j]]
            low = mean-stds[j][idx[j]]
            high = mean+stds[j][idx[j]]
            axes[j].plot(range(len(mean)), mean, colors[k], linewidth=1)
            axes[j].plot(range(len(mean)), low, colors[k], linewidth=1, alpha=0.3)
            axes[j].plot(range(len(mean)), high, colors[k], linewidth=1, alpha=0.3)
            axes[j].fill_between(range(dim.start, dim.stop, dim.step), low[dim], high[dim],
                                 facecolor=colors[k], alpha=0.2)
            # axes[j].errorbar(range(len(means[j][dim])), means[j][dim], yerr=[error_l[j][dim], error_u[j][dim]])
            # axes[j].errorbar(range(len(means[j][dim])), means[j][dim], yerr=stds[j][dim], capsize=5)
            # axes[j].set_ylim([-2, 2])
    for j in range(layer_num):
        axes[j].plot([0, len(means[0])], [0, 0], 'k', linewidth=1)


def parallel_coord(id_states, id_):

    import matplotlib.pyplot as plt

    stds, means, error_l, error_u, idx = compute_stats(id_states[id_], True)
    layer_num = len(means)
    fig, axes = plt.subplots(nrows=layer_num, sharex=True, figsize=(15, 9))

    states = id_states[id_]
    num = len(states)
    dim = slice(0, 600, 3)
    h_scale = range(600)
    for j in range(layer_num):
        for state in states:
            axes[j].plot(h_scale[dim], state[j][idx[j]][dim], colors[0], linewidth=1, alpha=(1.0/num)**0.8)

        axes[j].plot([0, len(means[0])], [0, 0], 'k', linewidth=1)


def scatter(id_states, ids, freqs):

    import matplotlib.pyplot as plt

    alphas = [(1.0/ freq)**0.8 for freq in freqs]
    stds, means, error_l, error_u, idx = compute_stats(id_states[ids[0]], True)
    layer_num = len(means)

    fig, axes = plt.subplots(nrows=layer_num, sharex=True, figsize=(15, 9))
    for k, id_ in enumerate(ids):
        states = id_states[id_]
        dim = slice(0, len(means[0]), 4)
        h_scale = range(0, len(means[0]), 4)
        for state in states:
            for j in range(layer_num):
                axes[j].plot(h_scale, state[j][idx[j]][dim], colors[k]+'.', alpha=alphas[k], linewidth=1)

    for j in range(layer_num):
        axes[j].plot([0, len(means[0])], [0, 0], 'k', linewidth=1)


if __name__ == '__main__':

    data_name = 'ptb'
    model_name = 'LSTM-PTB'
    state_name = 'state_c'
    words, state_diff = load_words_and_state(data_name, model_name, state_name)

    # embedding = get_model_params('./config/rnn.yml')
    #
    id_to_state = sort_by_id(words, state_diff)
    id_freq = [len(states) if states is not None else 0 for states in id_to_state]
    # state_shape = state_diff[0].shape
    # layer_num = state_shape[0]
    # mean_n = [[] for i in range(layer_num)]
    # std_n = [[] for i in range(layer_num)]
    # for i, states in enumerate(id_to_state):
    #     if states is None:
    #         stds = np.zeros(state_shape, dtype=np.float32)
    #         means = np.zeros(state_shape, dtype=np.float32)
    #     else:
    #         stds, means, error_l, error_u, idx = compute_stats(states, False)
    #     for j, mean in enumerate(means):
    #         mean_n[j].append(mean)
    #     for j, std in enumerate(stds):
    #         std_n[j].append(std)
    #
    # for i in range(layer_num):
    #     cand, mean, std = find_candidate(mean_n[i][:1000], std_n[i][:1000], 20)
    #     lists2csv(cand, '-'.join(['cand', data_name, model_name, str(i), str(20)]) + '.csv')
    #     lists2csv(mean, '-'.join(['mean', data_name, model_name, str(i), str(20)]) + '.csv')
    #     lists2csv(std, '-'.join(['std', data_name, model_name, str(i), str(20)]) + '.csv')

    import matplotlib.pyplot as plt

    # plot_words_states(id_to_state, [28, 163])
    # plot_words_states(id_to_state, [28, 14])
    # plot_words_states(id_to_state, [28, 17])

    print('id: {:d}, freq: {:d}'.format(28, id_freq[28]))
    print('id: {:d}, freq: {:d}'.format(1, id_freq[1]))
    print('id: {:d}, freq: {:d}'.format(14, id_freq[14]))
    # scatter(id_to_state, [28, 163], [id_freq[28], id_freq[163]])
    # plot_words_states(id_to_state, [28, 11])
    # plt.savefig('he-by.png', bbox_inches='tight')
    scatter(id_to_state, [28], [id_freq[28]])
    plt.savefig('he-scatter.png', bbox_inches='tight')
    parallel_coord(id_to_state, 28)
    plt.savefig('he-para-coord.png', bbox_inches='tight')

    plt.show(block=True)
    print("Done")


    # init_tf_environ(FLAGS.gpu_num)
    # datasets = get_datasets_by_name('ptb', ['test'])
    # test_data = datasets['test']

    # model, train_config = build_model('./config/lstm.yml', False)
    # model.add_evaluator(10, 1, 1)
    #
    # print('Preparing data')
    # producers = pour_data(train_config.dataset, ['test'], 10, 1)
    # inputs, targets, epoch_size = producers[0]
    # model.restore()
    #
    # model.run_with_context(model.evaluator.evaluate_and_record, inputs, targets,
    #                        StateRecorder(train_config.dataset, model.name), verbose=True)

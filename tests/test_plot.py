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


colors = ['#587EB6', '#C95A5F', '#63B075', 'r', 'm', 'y']


def plot_words_states(id_states, ids, percent=50):

    import matplotlib.pyplot as plt

    means, stds, error_l, error_u, idx = compute_stats(id_states[ids[0]], True)
    layer_num = len(means)

    fig, axes = plt.subplots(nrows=layer_num, sharex=True, figsize=(15, 9))
    for k, id_ in enumerate(ids):
        state = id_states[id_]
        means, stds, error_l, error_u, _ = compute_stats(state, False, percent)
        dim = slice(0, len(means[0]), 1)
        for j in range(layer_num):
            mean = means[j][idx[j]]
            low = mean-error_l[j][idx[j]]
            high = mean+error_u[j][idx[j]]
            axes[j].plot(range(len(mean)), mean, colors[k], linewidth=1, alpha=0.8)
            axes[j].plot(range(len(mean)), low, colors[k], linewidth=1, alpha=0.3)
            axes[j].plot(range(len(mean)), high, colors[k], linewidth=1, alpha=0.3)
            axes[j].fill_between(range(dim.start, dim.stop, dim.step), low[dim], high[dim],
                                 facecolor=colors[k], alpha=0.2)
            # axes[j].errorbar(range(len(means[j][dim])), means[j][dim], yerr=[error_l[j][dim], error_u[j][dim]])
            # axes[j].errorbar(range(len(means[j][dim])), means[j][dim], yerr=stds[j][dim], capsize=5)

    for j in range(layer_num):
        axes[j].plot([0, len(means[0])], [0, 0], 'k', linewidth=1)
        axes[j].set_ylim([-2.1 - 0.3 * j, 2.1 + 0.3 * j])


def parallel_coord(id_states, id_, salience=None, fields=None):

    import matplotlib.pyplot as plt

    means, stds, error_l, error_u, idx = compute_stats(id_states[id_], True)
    layer_num = len(means)
    fig, axes = plt.subplots(nrows=layer_num, sharex=True, figsize=(15, 9))

    states = id_states[id_]
    num = len(states)
    dim = slice(0, len(means[0]), 5)
    h_scale = range(len(means[0]))

    for j in range(layer_num):
        for state in states:
            axes[j].plot(h_scale[dim], state[j][idx[j]][dim], colors[0], linewidth=1, alpha=(1.0/num)**0.8)
        if salience is not None:
            _salience = salience[id_]
            for k, field in enumerate(fields):
                axes[j].plot(h_scale[dim], _salience[field][j][idx[j]][dim] / 4, colors[k+1], linewidth=1)

        axes[j].plot([0, len(means[0])], [0, 0], 'k', linewidth=1)
        axes[j].set_ylim([-2.1 - 0.3 * j, 2.1 + 0.3 * j])


def scatter(id_states, ids, freqs):

    import matplotlib.pyplot as plt

    alphas = [(2.0 / freq)**0.8 for freq in freqs]
    means, stds, error_l, error_u, idx = compute_stats(id_states[ids[0]], True)
    layer_num = len(means)

    fig, axes = plt.subplots(nrows=layer_num, sharex=True, figsize=(15, 9))
    for k, id_ in enumerate(ids):
        states = id_states[id_]
        dim = slice(0, len(means[0]), 5)
        h_scale = range(0, len(means[0]), 5)
        for state in states:
            for j in range(layer_num):
                axes[j].plot(h_scale, state[j][idx[j]][dim], colors[k], lw=0, marker='.', alpha=alphas[k])

    for j in range(layer_num):
        axes[j].plot([0, len(means[0])], [0, 0], 'k', linewidth=1)
        axes[j].set_ylim([-2.1 - 0.3 * j, 2.1 + 0.3 * j])


if __name__ == '__main__':

    data_name = 'ptb'
    model_name = 'LSTM-PTB'
    state_name = 'state_h'
    words, state_diff = load_words_and_state(data_name, model_name, state_name)

    # embedding = get_model_params('./config/rnn.yml')
    #
    id_to_state = sort_by_id(words, state_diff)
    id_freq = [len(states) if states is not None else 0 for states in id_to_state]

    import matplotlib.pyplot as plt

    if data_name == 'shakespeare':
        ####
        # SHAKESPEARE
        ####
        plot_words_states(id_to_state, [3, 28], 60)
        plt.savefig('and-but.png', bbox_inches='tight')

        parallel_coord(id_to_state, 25)
        plt.savefig('he-para-coord.png', bbox_inches='tight')

        parallel_coord(id_to_state, 20)
        plt.savefig('for-para-coord.png', bbox_inches='tight')

        # plot_words_states(id_to_state, [39, 423], 60)
        # plt.savefig('no-yes.png', bbox_inches='tight')

        # scatter(id_to_state, [20], [id_freq[20]])
        # plt.savefig('for-scatter.png', bbox_inches='tight')
        #
        # plot_words_states(id_to_state, [25, 57], 60)
        # plt.savefig('he-she.png', bbox_inches='tight')

        print("Done")

    if data_name == 'ptb' and False:
        #####
        ## PTB
        #####
        plot_words_states(id_to_state, [104], 60)
        plt.savefig('bank.png', bbox_inches='tight')

        scatter(id_to_state, [104], [id_freq[104]])
        plt.savefig('bank-scatter.png', bbox_inches='tight')

        parallel_coord(id_to_state, 104)
        plt.savefig('bank-para-coord.png', bbox_inches='tight')

        plot_words_states(id_to_state, [28, 11], 60)
        plt.savefig('he-for.png', bbox_inches='tight')

        plot_words_states(id_to_state, [28, 14], 60)
        plt.savefig('he-it.png', bbox_inches='tight')

        plot_words_states(id_to_state, [28, 163], 60)
        plt.savefig('he-she.png', bbox_inches='tight')

        plot_words_states(id_to_state, [28, 17], 60)
        plt.savefig('he-by.png', bbox_inches='tight')

        plot_words_states(id_to_state, [11, 17], 60)
        plt.savefig('for-by.png', bbox_inches='tight')

        plot_words_states(id_to_state, [11], 60)
        plt.savefig('for.png', bbox_inches='tight')

        scatter(id_to_state, [11], [id_freq[11]])
        plt.savefig('for-scatter.png', bbox_inches='tight')

        scatter(id_to_state, [28, 163], [id_freq[28], id_freq[163]])
        plt.savefig('he-she-scatter.png', bbox_inches='tight')

        scatter(id_to_state, [28], [id_freq[28]])
        plt.savefig('he-scatter.png', bbox_inches='tight')

        parallel_coord(id_to_state, 28)
        plt.savefig('he-para-coord.png', bbox_inches='tight')

        parallel_coord(id_to_state, 11)
        plt.savefig('for-para-coord.png', bbox_inches='tight')

        parallel_coord(id_to_state, 14)
        plt.savefig('by-para-coord.png', bbox_inches='tight')
        print("Done")

    # plt.show(block=True)

    ###
    # scripts loading salience
    ###
    with open("salience-1000-states.pkl", 'rb') as f:
        salience200 = pickle.loads(f.read())

    parallel_coord(id_to_state, 18, salience200, [state_name])
    plt.savefig('he-para-sa.png', bbox_inches='tight')

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

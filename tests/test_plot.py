"""
Tests the hidden state correlation
"""

import pickle

# import tensorflow as tf
import numpy as np

from rnnvis.procedures import build_model
from rnnvis.state_processor import load_words_and_state, sort_by_id
from rnnvis.plotter import scatter, parallel_coord, plot_words_states


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

    if data_name == 'ptb':
        #####
        ## PTB
        #####
        plot_words_states([id_to_state[i] for i in [104]], 60, labels=['bank'], save_path='bank.png')

        scatter([id_to_state[i] for i in [104]], labels=['bank'], save_path='bank-scatter.png')

        parallel_coord([id_to_state[i] for i in [104]], save_path='bank-para-coord.png')

        plot_words_states([id_to_state[i] for i in [11, 28]], 60, rank=1, labels=['for', 'he'],
                          save_path='he-for.png')

        plot_words_states([id_to_state[i] for i in [28, 14]], 60, labels=['he', 'it'], save_path='he-it.png')

        plot_words_states([id_to_state[i] for i in [28, 163]], 60, labels=['he', 'she'], save_path='he-she.png')

        plot_words_states([id_to_state[i] for i in [11, 17]], 60, labels=['for', 'by'], save_path='for-by.png')

        scatter([id_to_state[i] for i in [11]], labels=['for'], save_path='for-scatter.png')

        scatter([id_to_state[i] for i in [28]], labels=['he'], save_path='he-scatter.png')

        scatter([id_to_state[i] for i in [28, 163]], labels=['he', 'she'], save_path='he-she-scatter.png')

        parallel_coord([id_to_state[i] for i in [28]], save_path='he-para-coord.png')

        parallel_coord([id_to_state[i] for i in [11]], save_path='for-para-coord.png')

        parallel_coord([id_to_state[i] for i in [14]], save_path='by-para-coord.png')

        print("Done")

    # plt.show(block=True)

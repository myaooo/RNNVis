"""
Tests the hidden state correlation
"""

import pickle

# import tensorflow as tf
import numpy as np

from rnnvis.procedures import build_model
from rnnvis.state_processor import load_words_and_state, sort_by_id
from rnnvis.plotter import scatter, parallel_coord, plot_words_states
from rnnvis.db import get_dataset


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
    state_name = 'state_c'
    diff = False
    words, state_diff = load_words_and_state(data_name, model_name, state_name, diff)

    id_to_state = sort_by_id(words, state_diff)
    word_to_id = get_dataset(data_name, ['word_to_id'])['word_to_id']

    he = word_to_id['he']
    she = word_to_id['she']
    for_ = word_to_id['for']
    by = word_to_id['by']
    it = word_to_id['it']
    no = word_to_id['no']
    yes = word_to_id['yes']
    good = word_to_id['good']
    bad = word_to_id['bad']

    n_units = id_to_state[0][0].shape[1]
    if n_units < 150:
        every = 1
    else:
        every = n_units // 150 + 1

    if data_name == 'imdb' or data_name == 'imdb-small' or data_name == 'yelp-test':

        plot_words_states([id_to_state[i] for i in [he, she]], 60, labels=['he', 'she'], save_path='he-she.png')

        scatter([id_to_state[i] for i in [for_]], labels=['for'], every=every, save_path='for-scatter.png')

        parallel_coord([id_to_state[i] for i in [for_]], every, save_path='for-para-coord.png')

        plot_words_states([id_to_state[i] for i in [good, bad]], 60, labels=['good', 'bad'], save_path='good-bad.png')

        plot_words_states([id_to_state[i] for i in [yes, no]], 60, labels=['yea', 'no'], save_path='yes-no.png')

        plot_words_states([id_to_state[i] for i in [good, yes]], 60, labels=['good', 'yes'], save_path='good-yes.png')

    if data_name == 'ptb' or data_name == 'shakespeare':
        ####
        # PTB
        ####
        plot_words_states([id_to_state[i] for i in [104]], 60, labels=['bank'], save_path='bank.png')

        scatter([id_to_state[i] for i in [104]], labels=['bank'], every=every, save_path='bank-scatter.png')

        parallel_coord([id_to_state[i] for i in [104]], every, save_path='bank-para-coord.png')

        plot_words_states([id_to_state[i] for i in [for_, he]], 60, rank=1, labels=['for', 'he'],
                          save_path='he-for.png')

        plot_words_states([id_to_state[i] for i in [he, it]], 60, labels=['he', 'it'], save_path='he-it.png')

        plot_words_states([id_to_state[i] for i in [he, she]], 60, labels=['he', 'she'], save_path='he-she.png')

        plot_words_states([id_to_state[i] for i in [for_, by]], 60, labels=['for', 'by'], save_path='for-by.png')

        scatter([id_to_state[i] for i in [for_]], labels=['for'], every=every, save_path='for-scatter.png')

        scatter([id_to_state[i] for i in [he]], labels=['he'], every=every, save_path='he-scatter.png')

        scatter([id_to_state[i] for i in [he, she]], labels=['he', 'she'], every=every, save_path='he-she-scatter.png')

        parallel_coord([id_to_state[i] for i in [he]], every, save_path='he-para-coord.png')

        parallel_coord([id_to_state[i] for i in [for_]], every, save_path='for-para-coord.png')

        parallel_coord([id_to_state[i] for i in [by]], every, save_path='by-para-coord.png')

    if state_name == 'state_c' or state_name == 'state_h':

        state_name2 = 'state_h' if state_name == 'state_c' else 'state_c'
        state_name2 = 'nondiff' if diff else 'diff'
        _, state2_diff = load_words_and_state(data_name, model_name, state_name, not diff)
        id_to_state2 = sort_by_id(words, state2_diff)

        plot_words_states([id_to_state[he], id_to_state2[he]], 60, [state_name, state_name2], save_path='he-he.png')

        plot_words_states([id_to_state[for_], id_to_state2[for_]], 60, [state_name, state_name2],
                          save_path='for-for.png')

        plot_words_states([id_to_state[it], id_to_state2[it]], 60, [state_name, state_name2],
                          save_path='it-it.png')

        plot_words_states([id_to_state[good], id_to_state2[good]], 60, [state_name, state_name2],
                          save_path='good-good.png')


    print("Done")

    # plt.show(block=True)

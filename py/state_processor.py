"""
Helpers that deal with all the computations related to hidden states
"""

import pickle

import numpy as np

from py.db.language_model import query_evals, query_evaluation_records
from py.utils.io_utils import file_exists, get_path


def cal_diff(arrays):
    diff_arrays = []
    for i in range(len(arrays)-1):
        diff_arrays.append(arrays[i+1] - arrays[i])
    return diff_arrays


def cal_similar1(array):
    """
    :param array: 2D [n_state, n_words], each row as a states history
    :return: a matrix of n_state x n_state measuring the similarity
    """
    return array * array.T


def fetch_state_of_eval(eval_id, field_name='state_c', diff=True):
    records = query_evaluation_records(eval_id)
    word_ids = [record['word_id'] for record in records]
    if isinstance(field_name, list):
        assert isinstance(diff, list)
    else:
        field_name = [field_name]
        diff = [diff]
    states = []
    for i, field in enumerate(field_name):
        state = [record[field] for record in records]
        if diff[i]:
            state = [state[0]] + cal_diff(state)
        states.append(state)
    if len(states) == 1:
        states = states[0]
    return word_ids, states


def fetch_states(data_name, model_name, field_name='state_c', diff=True):
    evals = query_evals(data_name, model_name)
    if evals is None:
        raise LookupError("No eval records with data_name: {:s} and model_name: {:s}".format(data_name, model_name))
    word_ids = []
    states = []
    for eval in evals:
        word_ids_, states_ = fetch_state_of_eval(eval['_id'], field_name, diff)
        word_ids += word_ids_
        states += states_
    return word_ids, states


def sort_by_id(word_ids, states):
    max_id = max(word_ids)
    id_to_states = [None] * (max_id+1)
    for k, id_ in enumerate(word_ids):
        if id_to_states[id_] is None:
            id_to_states[id_] = []
        id_to_states[id_].append(states[k])
    return id_to_states


def compute_stats(states, sort_by_mean=True):
    layer_num = states[0].shape[0]
    states_layer_wise = []
    stds = []
    means = []
    errors_l = []
    errors_u = []
    indices = []
    idx = None
    for layer in range(layer_num):
        state_list = [state[layer] for state in states]
        states_mat = np.vstack(state_list)
        std = np.std(states_mat, axis=0)
        mean = np.mean(states_mat, axis=0)
        error_l = mean-np.min(states_mat, axis=0)
        error_u = np.max(states_mat, axis=0)-mean
        if sort_by_mean:
            idx = np.argsort(mean)
            mean = mean[idx]
            std = std[idx]
            error_u = error_u[idx]
            error_l = error_l[idx]
        indices.append(idx)
        stds.append(std)
        means.append(mean)
        errors_l.append(error_l)
        errors_u.append(error_u)
        states_layer_wise.append(states_mat)
    return stds, means, errors_l, errors_u, indices


def load_words_and_state(data_name, model_name, state_name, diff=True):
    word_file = data_name + '-' + model_name + '-words.pkl'
    states_file = data_name + '-' + model_name + '-' + state_name + ('-diff' if diff else '') + '.pkl'
    if file_exists(word_file) and file_exists(states_file):
        with open(word_file, 'rb') as f:
            words = pickle.loads(f.read())
        with open(states_file, 'rb') as f:
            states = pickle.loads(f.read())
    else:
        words, states = fetch_states(data_name, model_name, state_name, diff)
        with open(word_file, 'wb') as f:
            pickle.dump(words, f)
        with open(states_file, 'wb') as f:
            pickle.dump(states, f)
    return words, states


if __name__ == '__main__':
    data_name = 'ptb'
    model_name = 'LSTM-PTB'
    state_name = 'state_c'
    words, states = load_words_and_state(data_name, model_name, state_name, diff=False)
    states2 = [state[1, :] for state in states]
    states_mat = np.vstack(states2).T
    sim1 = cal_similar1(states_mat)
    cov = np.cov(states_mat)
    sims = [sim1, cov]
    sim1_path = '-'.join([data_name, model_name, state_name, '2', 'sim1']) + '.pkl'
    sim2_path = '-'.join([data_name, model_name, state_name, '2', 'cov']) + '.pkl'
    for i, path in enumerate([sim1_path, sim2_path]):
        with open(get_path(path)) as f:
            pickle.dump(sims[i], f)

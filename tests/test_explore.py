"""
Tests the hidden state correlation
"""

import pickle

# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from py.procedures import build_model, init_tf_environ, pour_data
from py.rnn.evaluator import StateRecorder
from py.db.language_model import query_evals, query_evaluation_records
from py.utils.io_utils import file_exists

# flags = tf.flags
# flags.DEFINE_integer('gpu_num', 0, "The number of the gpu to use, 0 to use no gpu.")
# FLAGS = flags.FLAGS

#
# def config_path():
#     return FLAGS.config_path


def fetch_states_of_eval(eval_id):
    records = query_evaluation_records(eval_id)
    word_ids = [record['word_id'] for record in records]
    state_c = [record['state_c'] for record in records]
    # state_h = [record['state_h'] for record in records]
    state_c_diff = [state_c[0]]
    for i in range(len(state_c)-1):
        state_c_diff.append(state_c[i+1] - state_c[i])
    return word_ids, state_c_diff


def fetch_states(data_name, model_name):
    evals = query_evals(data_name, model_name)
    word_ids = []
    state_c_diff = []
    for eval in evals:
        eval_id = eval['_id']
        word_ids_, state_c_diff_ = fetch_states_of_eval(eval_id)
        word_ids += word_ids_
        state_c_diff += state_c_diff_
    return word_ids, state_c_diff


def sort_by_id(word_ids, states):
    max_id = max(word_ids)
    id_to_states = [None] * (max_id+1)
    for k, id_ in enumerate(word_ids):
        if id_to_states[id_] is None:
            id_to_states[id_] = []
        id_to_states[id_].append(states[k])
    return id_to_states


def compute_stats(states):
    layer_num = states[0].shape[0]
    states_layer_wise = []
    stds = []
    means = []
    error_l = []
    error_u = []
    for layer in range(layer_num):
        state_list = [state[layer] for state in states]
        states_mat = np.vstack(state_list)
        stds.append(np.std(states_mat, axis=0))
        mean = np.mean(states_mat, axis=0)
        means.append(mean)
        error_l.append(mean-np.min(states_mat, axis=0))
        error_u.append(np.max(states_mat, axis=0)-mean)
        states_layer_wise.append(states_mat)
    return stds, means, error_l, error_u


if __name__ == '__main__':

    if file_exists('words.pkl') and file_exists('states.pkl'):
        with open('words.pkl', 'rb') as f:
            words = pickle.loads(f.read())
        with open('states.pkl', 'rb') as f:
            state_diff = pickle.loads(f.read())
    else:
        words, state_diff = fetch_states('ptb', 'LSTM-PTB')
        with open('words.pkl', 'wb') as f:
            pickle.dump(words, f)
        with open('states.pkl', 'wb') as f:
            pickle.dump(state_diff, f)

    id_to_state = sort_by_id(words, state_diff)

    for i, states in enumerate(id_to_state):
        stds, means, error_l, error_u = compute_stats(states)
        fig, axes = plt.subplots(nrows=4, sharex=True, figsize=(9, 9))
        dim = slice(0, 100)
        for j in range(2):
            axes[j*2].errorbar(range(len(means[j][dim])), means[j][dim], yerr=stds[j][dim])
            axes[j*2+1].errorbar(range(len(means[j][dim])), means[j][dim], yerr=[error_l[j][dim], error_u[j][dim]])

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

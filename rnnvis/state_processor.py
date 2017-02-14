"""
Helpers that deal with all the computations related to hidden states
"""

import pickle

import numpy as np

import matplotlib.pyplot as plt


from rnnvis.db import get_dataset
from rnnvis.db.language_model import query_evals, query_evaluation_records
from rnnvis.utils.io_utils import file_exists, get_path


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
    return np.dot(array, array.T)


def normalize(array):
    max_ = np.max(array)
    min_ = np.min(array)
    return (array - min_) / (max_ - min_)


def sigmoid(array):
    return 1 / (1 + np.exp(-array))


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
    """
    Fetch the word_ids and states of the eval records by data_name and model_name
    :param data_name:
    :param model_name:
    :param field_name: the name of the desired state, can be a list of fields
    :param diff: True if you want the diff, should also be list when field_name is a list
    :return: a pair (word_id, states)
    """
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


def fetch_freq_words(data_name, k=100):
    id_to_word = get_dataset(data_name, ['id_to_word'])['id_to_word']
    return id_to_word[:k]


def load_words_and_state(data_name, model_name, state_name, diff=True):
    word_file = data_name + '-' + model_name + '-words.pkl'
    states_file = data_name + '-' + model_name + '-' + state_name + ('-diff' if diff else '') + '.pkl'
    if file_exists(word_file) and file_exists(states_file):
        with open(get_path('_cached', word_file), 'rb') as f:
            words = pickle.loads(f.read())
        with open(get_path('_cached', states_file), 'rb') as f:
            states = pickle.loads(f.read())
    else:
        words, states = fetch_states(data_name, model_name, state_name, diff)
        with open(get_path('_cached', word_file), 'wb') as f:
            pickle.dump(words, f)
        with open(get_path('_cached', states_file), 'wb') as f:
            pickle.dump(states, f)
    return words, states


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, data, extent=None, interval=5):
        from matplotlib import animation
        # self.stream = self.data_stream()
        self.extent = extent
        self.data = data
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=interval,
                                           init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        xy = self.data[0]
        self.scat = self.ax.scatter(xy[:, 0], xy[:, 1], 10, animated=True)
        if self.extent is not None:
            self.ax.axis(self.extent)

        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    # def data_stream(self):
    #     """Generate a random walk (brownian motion). Data is scaled to produce
    #     a soft "flickering" effect."""
    #     len
    #     for data in self.data:
    #         yield data
        # data = np.random.random((4, self.numpoints))
        # xy = data[:2, :]
        # s, c = data[2:, :]
        # xy -= 0.5
        # xy *= 10
        # while True:
        #     xy += 0.03 * (np.random.random((2, self.numpoints)) - 0.5)
        #     s += 0.05 * (np.random.random(self.numpoints) - 0.5)
        #     c += 0.02 * (np.random.random(self.numpoints) - 0.5)
        #     yield data

    def update(self, i):
        """Update the scatter plot."""
        data = self.data[i % len(self.data)]

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        # self.scat._sizes = 300 * abs(data[2])**1.5 + 100
        # # Set colors..
        # self.scat.set_array(data[3])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def show(self):
        plt.show()


if __name__ == '__main__':
    data_name = 'ptb'
    model_name = 'LSTM-PTB'
    state_name = 'state_c'
    print('loading states...')

    # print('calculating similarity')
    # sim1 = cal_similar1(states_mat)
    # cov = np.cov(states_mat)
    # sims = [sim1, cov, sigmoid(sim1/100000)]
    sim1_path = '-'.join([data_name, model_name, state_name, '2', 'sim1']) + '.json'
    sim2_path = '-'.join([data_name, model_name, state_name, '2', 'cov']) + '.json'
    sim3_path = '-'.join([data_name, model_name, state_name, '2', 'sim1-sigmoid']) + '.json'
    # print("max: {:f}, min: {:f}".format(np.max(sim1), np.min(sim1)))

    # from rnnvis.utils.io_utils import dict2json
    #
    # for i, path in enumerate([sim1_path, sim2_path, sim3_path]):
    #     dict2json(sims[i].tolist(), path)

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    # axes[0].imshow(sims[0], extent=[0, 600, 0, 600])
    # axes[1].imshow(sims[2], extent=[0, 600, 0, 600])
    # plt.show(block=True)

    if file_exists('sample5000.pkl'):
        print("sampling")
        with open('sample5000.pkl', 'rb') as f:
            sample = pickle.load(f)
    else:
        words, states = load_words_and_state(data_name, model_name, state_name, diff=False)
        states2 = [state[1, :] for state in states]
        states_mat = np.vstack(states2).T
        print("sampling")
        sample_idx = np.random.randint(0, states_mat.shape[1], 5000)
        sample = states_mat[:, sample_idx]
        with open('sample5000.pkl', 'wb') as f:
            pickle.dump(sample, f)

    # import json
    # with open(sim1_path) as f:
    #     sample = np.array(json.load(f))

    import rnnvis.vendor.tsne as tsne

    sample = tsne.pca(sample/100, 50).real
    print("doing tsne")
    projected = tsne.tsne(sample, 2, 50, 30.0, 1000)

    # base = np.random.random((10, 2))*10
    # projected = [base]
    # for i in range(100):
    #     projected.append(projected[i]+np.random.random((10,2))*0.5 - 0.25)
    min_x = min([np.min(data[:, 0]) for data in projected])
    min_y = min([np.min(data[:, 1]) for data in projected])
    max_x = max([np.max(data[:, 0]) for data in projected])
    max_y = max([np.max(data[:, 1]) for data in projected])
    anim = AnimatedScatter(projected, [min_x, max_x, min_y, max_y], 50)
    anim.show()


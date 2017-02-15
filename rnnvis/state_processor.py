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
    def __init__(self, data, figsize=None, interval=5):
        from matplotlib import animation
        # self.stream = self.data_stream()
        self.figsize = [6, 6] if figsize is None else figsize
        self.data = data
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.setup_plot()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, len(data), interval=interval, repeat_delay=1000,
                                           blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        xy = self.data[0]
        self.scat = self.ax.scatter(xy[:, 0], xy[:, 1], 8, xy[:, 2], animated=True, alpha=0.5)
        self.ax.axis([-self.figsize[0]/2.0, self.figsize[0]/2.0, -self.figsize[1]/2.0, self.figsize[1]/2.0])

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
        data = self.data[i]
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
        # Set sizes...
        # self.scat._sizes = 300 * abs(data[2])**1.5 + 100
        # # Set colors..
        # self.scat.set_array(data[:, 2])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def show(self):
        plt.show()

    def save(self, filename, fps=15, bitrate=1800):
        from matplotlib import animation
        _writer = animation.writers['ffmpeg']
        writer = _writer(fps=fps, metadata=dict(artist='Ming'), bitrate=bitrate)
        self.ani.save(filename, writer)




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


    import rnnvis.vendor.tsne as tsne

    # for layer 2
    # if file_exists('sample5000pca50.pkl'):
    #     print("sampling")
    #     with open('sample5000pca50.pkl', 'rb') as f:
    #         sample = pickle.load(f)
    # else:
    #     words, states = load_words_and_state(data_name, model_name, state_name, diff=False)
    #     states2 = [state[1, :] for state in states]
    #     states_mat = np.vstack(states2).T
    #     print("sampling")
    #     sample_idx = np.random.randint(0, states_mat.shape[1], 5000)
    #     sample = states_mat[:, sample_idx]
    #     sample = tsne.pca(sample / 100, 50).real
    #     with open('sample5000pca50.pkl', 'wb') as f:
    #         pickle.dump(sample, f)

    # for both layer
    if file_exists('sample5000pca50-2.pkl'):
        print("sampling")
        with open('sample5000pca50-2.pkl', 'rb') as f:
            sample = pickle.load(f)
    else:
        words, states = load_words_and_state(data_name, model_name, state_name, diff=False)
        states1 = [state[0, :] for state in states]
        states2 = [state[1, :] for state in states]
        states_mat = np.hstack([np.vstack(states1), np.vstack(states2)]).T
        print("sampling")
        sample_idx = np.random.randint(0, states_mat.shape[1], 5000)
        sample = states_mat[:, sample_idx]
        sample = tsne.pca(sample / 100, 50).real
        with open('sample5000pca50-2.pkl', 'wb') as f:
            pickle.dump(sample, f)

    # import json
    # with open(sim1_path) as f:
    #     sample = np.array(json.load(f))


    print("doing tsne")
    projected = tsne.tsne(sample, 2, 50, 40.0, 1000)

    # base = np.random.random((10, 2))*1.0
    # projected = [base]
    # for i in range(100):
    #     projected.append(projected[i]+np.random.random((10,2))*0.5 - 0.25)

    points_num = projected[0].shape[0]
    color = np.ones((points_num, 1), dtype=np.float32)
    color[:points_num//2, :] -= 2
    projected = [np.hstack((project, color)) for project in projected]

    anim = AnimatedScatter(projected, [6, 6], 50)

    anim.save('test.mp4')
    # anim.show()

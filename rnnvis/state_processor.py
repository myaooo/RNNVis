"""
Helpers that deal with all the computations related to hidden states
For example usage, see the main function below
"""

import pickle

import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


from rnnvis.db import get_dataset
from rnnvis.db.db_helper import query_evals, query_evaluation_records
from rnnvis.utils.io_utils import file_exists, get_path, dict2json
from rnnvis.vendor import tsne, mds


def cal_diff(arrays):
    """
    Given a list of same shape ndarray or an ndarray,
    calculate the difference a_t - a_{t-1} along the axis 0
    :param arrays: a list of same shaped ndarray or an ndarray
    :return: a list of diff
    """
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
    Fetch the word_ids and states of the eval records by data_name and model_name from db
    :param data_name:
    :param model_name:
    :param field_name: the name of the desired state, can be a list of fields
    :param diff: True if you want the diff, should also be list when field_name is a list
    :return: a pair (word_id, states)
    """
    evals = query_evals(data_name, model_name)
    if evals.count() == 0:
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


def compute_stats(states, sort_by_mean=True, percent=50):
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
        # error_l = mean-np.min(states_mat, axis=0)
        # error_u = np.max(states_mat, axis=0)-mean
        error_l = mean - np.percentile(states_mat, (100-percent)/2, axis=0)
        error_u = np.percentile(states_mat, 50 + percent/2, axis=0) - mean
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
    return means, stds, errors_l, errors_u, indices


def get_empirical_strength(id_states, strength_func):
    """

    :param id_states: a list, with each
    :param strength_func: np.mean, etc
    :return:
    """
    state_shape = id_states[0][0].shape

    def strenth_map(states):
        if states is None:
            return np.zeros(state_shape)
        states_mat = np.stack(states, axis=0)
        return strength_func(states_mat)

    strength_list = list(map(strenth_map, id_states))
    return strength_list


def strength2json(strength_list, words, labels=None, path=None):
    """

    :param strength_list: a list of ndarray (n_layer, n_states)
    :param words: word (str) for each strength
    :param labels: additional labels
    :param path: saving path
    :return:
    """
    if labels is None:
        labels = [0] * len(strength_list)
    points = [{'word': words[i], 'strength': strength.tolist(), 'label': labels[i]}
              for i, strength in enumerate(strength_list)]
    return dict2json(points, path)


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


def get_state_signature(data_name, model_name, state_name, layer=None, sample_size=5000, dim=50):
    """

    :param data_name: str
    :param model_name: str
    :param state_name: str
    :param layer: start from 0
    :param sample_size:
    :param dim:
    :return:
    """
    layer_str = 'all' if layer is None else ''.join([str(l) for l in layer])
    file_name = '-'.join([data_name, model_name, state_name, 'all' if layer is None else layer_str,
                          str(sample_size), str(dim) if dim is not None else str(sample_size)]) + '.pkl'
    if file_exists(file_name):
        print("sampling")
        with open(file_name, 'rb') as f:
            sample = pickle.load(f)
    else:
        words, states = load_words_and_state(data_name, model_name, state_name, diff=False)
        layer = layer if layer is not None else list(range(states[0].shape[0]))
        state_layers = []
        for l in layer:
            state_layers.append([state[l, :] for state in states])
        states_mat = np.hstack(state_layers).T
        print("sampling")
        sample_idx = np.random.randint(0, states_mat.shape[1], sample_size)
        sample = states_mat[:, sample_idx]
        if dim is not None:
            print("doing PCA...")
            sample, variance = tsne.pca(sample, dim)
            print("PCA kept {:f}% of variance".format(variance*100))
        with open(file_name, 'wb') as f:
            pickle.dump(sample, f)
    return sample


def tsne_project(data, perplexity, init_dim=50, lr=50, max_iter=1000):
    """
    Do t-SNE projection with given configuration
    :param data: 2D numpy.ndarray of shape [n_data, feature_dim]
    :param perplexity:
    :param init_dim: in case feature size too large, do PCA to reduce feature dim if needed
    :param lr: learning rate
    :param max_iter: the max iterations to run
    :return: the best solution in the run
    """
    _tsne_solver = tsne.TSNE(2, perplexity, lr)
    _tsne_solver.set_inputs(data, init_dim)
    _tsne_solver.run(max_iter)
    return _tsne_solver.get_best_solution()


def solution2json(solution, states_num, labels=None, path=None):
    """
    Save the solution to json file
    :param solution:
    :param states_num: a list specifying number of states in each layer, should add up the the solution size
    :param labels: additional labels for each states
    :param path:
    :return:
    """
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    if labels is None:
        labels = [0] * len(solution)
    layers = []
    state_ids = []
    for i, num in enumerate(states_num):
        layers += [i+1] * num
        state_ids += range(num)
    points = [{'coords': s, 'layer': layers[i], 'state_id': state_ids[i], 'label': labels[i]}
              for i, s in enumerate(solution)]
    return dict2json(points, path)


color_scheme = [[201/255, 90/255, 95/255], [88/255, 126/255, 182/255],
                [114/255, 189/255, 210/255], [99/255, 176/255, 117/255]]


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
        color_list.append(np.tile(np.array(color_scheme[i], np.float32), (num, 1)))
    color = np.vstack(color_list)

    projected = [np.hstack((project, color)) for project in projected]
    anim = AnimatedScatter(projected, [6, 6], 50)

    if path is not None:
        anim.save(path)
    anim.show()


def get_state_value(states, layer, dim):
    """
    Given the loaded states from load_words_and_states, a layer no. and a dim no.,
    return values of a specific state as a list
    :param states:
    :param layer:
    :param dim:
    :return:
    """
    return [state[layer, dim] for state in states]


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


if __name__ == '__main__':

    data_name = 'ptb'
    model_name = 'GRU-PTB'
    state_name = 'state'
    ###
    # Scripts that run tsne on states and produce .json file for front-end rendering
    ###
    print('loading states...')

    sample = get_state_signature(data_name, model_name, state_name, [1], 5000, 50)/10

    solution = tsne_project(sample, 40.0, 50, 50)
    labels = ([1] * (solution.shape[0])) # + ([0] * (solution.shape[0] // 2))
    solution2json(solution, [0, 600], labels, get_path('_cached', 'gru-state-tsne.json'))
    print("tsne saved")

    # scripts that run t-sne animation
    ###
    # print('loading states...')
    #
    # sample = get_state_signature(data_name, model_name, state_name, None, 5000, 50)/10
    # seed = (np.random.rand(30) + 5) * 2
    # sample = np.vstack([
    #     np.random.rand(100, 30) + seed,
    #     np.random.rand(100, 30) - seed,
    # ])
    #
    # create_animated_tsne(sample, 40.0, [600,600], init_dim=50, lr=50, max_iter=1000, path='test.mp4')

    ###
    # Scripts that calculate the mean
    ###
    words, states = load_words_and_state(data_name, model_name, state_name, diff=True)
    id_to_states = sort_by_id(words, states)
    strength_mat = get_empirical_strength(id_to_states[:200], lambda state_mat: np.mean(state_mat, axis=0)[1:, :])
    id_to_word = get_dataset(data_name, ['id_to_word'])['id_to_word']
    word_list = id_to_word[:200]
    strength2json(strength_mat, word_list, path=get_path('_cached', 'gru-state-strength.json'))

    ###
    # scripts performing mds
    ###
    # sample = get_state_signature(data_name, model_name, state_name, [1], 5000, None)
    # dist = squareform(pdist(sample, 'euclidean'))
    # y, eigs = mds.mds(dist)
    #
    # color = np.vstack([
    #     np.tile(np.array(color_scheme[0], np.float32), (600, 1))
    # ])
    # fig, ax = plt.subplots(figsize=[6, 6])
    # ax.scatter(y[:600, 0], y[:600, 1], 8, c=color[:600, :])
    # plt.show()


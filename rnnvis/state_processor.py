"""
Helpers that deal with all the computations related to hidden states
For example usage, see the main function below
"""

import pickle
from functools import lru_cache
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial.distance import pdist, squareform

from rnnvis.db import get_dataset
from rnnvis.db.db_helper import query_evals, query_evaluation_records, get_datasets_by_name
from rnnvis.utils.io_utils import file_exists, get_path, dict2json, before_save
from rnnvis.vendor import tsne, mds

_tmp_dir = '_cached/tmp'

#############
# Major APIs that the server calls
#############


@lru_cache(maxsize=32)
def get_empirical_strength(data_name, model_name, state_name, layer=-1, top_k=100):
    """
    A helper function that wraps cal_empirical_strength and cached the results in .pkl file for latter use
    :param data_name:
    :param model_name:
    :param state_name:
    :param layer: specify a layer, start from 0
    :param top_k: get the strength of the top k frequent words
    :return: a list of strength mat (np.ndarray) of shape [len(layer), state_size]
    """
    if not isinstance(layer, list):
        layer = [layer]
    if top_k > 1000:
        raise ValueError("selected words range too large, only support top 1000 frequent words!")
    top = 100 if top_k <= 100 else 500 if top_k <= 500 else 1000
    tmp_file = '-'.join([data_name, model_name, 'strength', state_name, str(top)]) + '.pkl'
    tmp_file = get_path(_tmp_dir, tmp_file)

    def cal_fn():
        # words, states = load_words_and_state(data_name, model_name, state_name, diff=True)
        id_to_states = load_sorted_words_states(data_name, model_name, state_name, diff=True)
        return cal_empirical_strength(id_to_states[:top], lambda state_mat: np.mean(state_mat, axis=0))

    id_strengths = maybe_calculate(tmp_file, cal_fn)

    return [id_strengths[i][layer] for i in range(top_k)]


@lru_cache(maxsize=128)
def get_an_empirical_strength(data_name, model_name, state_name, layer, k):
    id_to_states = load_sorted_words_states(data_name, model_name, state_name, diff=True)
    strength_list = cal_empirical_strength([id_to_states[k]], lambda state_mat: np.mean(state_mat, axis=0))
    strength = strength_list[0]
    if np.max(np.abs(strength)) > 1e-8:
        return strength[layer]
    return None


def strength2json(strength_list, words, labels=None, path=None):
    """
    A helper function that convert the results of get_empirical_strength
        to standard format to serve the web request.
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


@lru_cache(maxsize=32)
def get_state_signature(data_name, model_name, state_name, layer=None, sample_size=5000, dim=50):
    """
    A helper function that sampled the states records,
        and maybe do PCA (if `sample size` is different from `dim`).
        The results will be cached on disk.
    :param data_name: str
    :param model_name: str
    :param state_name: str
    :param layer: start from 0
    :param sample_size:
    :param dim:
    :return:
    """
    if layer is not None:
        if not isinstance(layer, list):
            layer = [layer]
    layer_str = 'all' if layer is None else ''.join([str(l) for l in layer])
    file_name = '-'.join([data_name, model_name, state_name, 'all' if layer is None else layer_str,
                          str(sample_size), str(dim) if dim is not None else str(sample_size)]) + '.pkl'
    file_name = get_path(_tmp_dir, file_name)

    def cal_fn(layers):
        words, states = load_words_and_state(data_name, model_name, state_name, diff=False)
        layers = layers if layers is not None else list(range(states[0].shape[0]))
        state_layers = []
        for l in layers:
            state_layers.append([state[l, :] for state in states])
        states_mat = np.hstack(state_layers).T
        print("sampling")
        sample_idx = np.random.randint(0, states_mat.shape[1], sample_size)
        sample = states_mat[:, sample_idx]
        if dim is not None:
            print("doing PCA...")
            sample, variance = tsne.pca(sample, dim)
            print("PCA kept {:f}% of variance".format(variance * 100))
        return sample

    return maybe_calculate(file_name, cal_fn, layer)


def get_tsne_projection(data_name, model_name, state_name, layer=-1, sample_size=5000, dim=50, perplexity=40.0):
    """
    A helper function that wraps get_state_signature and tsne_project,
        the results will be chached on disk for latter use.
    :param data_name:
    :param model_name:
    :param state_name:
    :param layer:
    :param sample_size:
    :param dim:
    :param perplexity:
    :return:
    """
    assert isinstance(layer, int), "tsne projection of only one layer is reasonable"
    tmp_file = '-'.join([data_name, model_name, state_name, 'tsne',
                         str(layer), str(dim), str(int(perplexity))]) + '.pkl'
    tmp_file = get_path(_tmp_dir, tmp_file)

    def cal_fn():
        sample = get_state_signature(data_name, model_name, state_name, layer, sample_size, dim) / 50
        print('Start doing t-SNE...')
        return tsne_project(sample, perplexity, dim, lr=50)

    tsne_solution = maybe_calculate(tmp_file, cal_fn)
    return tsne_solution


def solution2json(solution, states_num, labels=None, path=None):
    """
    Convert the tsne solution to json format
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
        state_ids += list(range(num))
    points = [{'coords': s, 'layer': layers[i], 'state_id': state_ids[i], 'label': labels[i]}
              for i, s in enumerate(solution)]
    return dict2json(points, path)


@lru_cache(maxsize=32)
def load_words_and_state(data_name, model_name, state_name, diff=True):
    """
    A wrapper function that wraps fetch_states and cached the results in .pkl file for latter use
    :param data_name:
    :param model_name:
    :param state_name:
    :param diff:
    :return: a pair of two list (word_list, states_list)
    """

    states_file = data_name + '-' + model_name + '-' + 'words' + '-' + state_name + ('-diff' if diff else '') + '.pkl'
    states_file = get_path(_tmp_dir, states_file)

    def cal_fn():
        return fetch_states(data_name, model_name, state_name, diff)

    words, states = maybe_calculate(states_file, cal_fn)
    return words, states


@lru_cache(maxsize=32)
def load_sorted_words_states(data_name, model_name, state_name, diff=True):
    """
    A wrapper function that wraps fetch_states and sort them according to ids,
        and cached the results in .pkl file for latter use
    :param data_name:
    :param model_name:
    :param state_name:
    :param diff:
    :return: a pair of two list (word_list, states_list)
    """
    states_file = '-'.join([data_name, model_name, 'words', state_name, 'sorted']) + ('-diff' if diff else '') + '.pkl'
    states_file = get_path(_tmp_dir, states_file)

    def cal_fn():
        words, states = fetch_states(data_name, model_name, state_name, diff)
        return sort_by_id(words, states)

    id_states = maybe_calculate(states_file, cal_fn)
    return id_states


@lru_cache(maxsize=32)
def get_state_statistics(data_name, model_name, state_name, diff=True, layer=-1, top_k=500, k=None):
    """
    Get state statistics, i.e. states mean reaction, 25~75 reaction range, 9~91 reaction range regarding top_k words
    :param data_name:
    :param model_name:
    :param state_name:
    :param diff:
    :param layer:
    :param top_k:
    :return: a dict containing statistics:
        {
            'mean': [top_k, n_states],
            'low1': [top_k, n_states], 25%
            'high1': [top_k, n_states], 75%
            'low2': [top_k, n_states], 9%
            'high2': [top_k, n_states], 91%
            'sort_idx': [top_k, n_states], each row represents sorted idx of mean reaction of states w.r.t. a word
            'freqs': [top_k,] frequency of each of the top_k words,
            'words': [top_k,], a list of words.
        }
    """
    if k is not None:
        # top_k = top_k if top_k > k else k
        start = (k // 100) * 100
        end = (k // 100 + 1) * 100
    else:
        start = 0
        end = 100 if top_k <= 100 else 500 if top_k <= 500 else 1000
    cal_range = range(start, end)

    tmp_file = '-'.join([data_name, model_name, state_name, 'statistics', str(start), str(end)]) \
               + ('-diff' if diff else '') + '.pkl'
    tmp_file = get_path(_tmp_dir, tmp_file)

    def cal_fn(data_name_, model_name_, state_name_, diff_, range_):
        # _words, states = load_words_and_state(data_name_, model_name_, state_name_, diff_)
        id_to_states = load_sorted_words_states(data_name_, model_name_, state_name_, diff_)
        _words = get_datasets_by_name(data_name_, ['id_to_word'])['id_to_word']
        words = []
        state_shape = id_to_states[0][0].shape
        dtype = id_to_states[0][0].dtype
        layer_num = state_shape[0]
        stats_list = []
        for i in range_:
            id_to_state = id_to_states[i]
            if id_to_state is None:  # some words may be seen in test set
                states = [np.zeros(state_shape, dtype)]  # use zeros as placeholder
            else:
                states = id_to_state
            stats_list.append(cal_state_statistics(states))
            words.append(_words[i])
        stats_layer_wise = []
        for layer_ in range(layer_num):
            stats = {}
            for field in stats_list[0][layer_].keys():
                value = np.vstack([stat[layer_][field] for stat in stats_list])
                stats[field] = value
            stats['freqs'] = np.array([len(id_state) if id_state is not None else 0 for id_state in id_to_states])
            stats_layer_wise.append(stats)
        return stats_layer_wise, words

    layer_wise_stats, words = maybe_calculate(tmp_file, cal_fn, data_name, model_name, state_name, diff, cal_range)
    stats = layer_wise_stats[layer]
    if k is None:
        # stats = {key: value[:(top_k)].tolist() for key, value in stats.items()}
        results = defaultdict(list)
        for i in range(end):
            if len(results['freqs']) == top_k:
                break
            if stats['freqs'][i] == 0:
                continue
            for key, value in stats.items():
                results[key].append(value[i].tolist())
            results['words'].append(words[i])

    else:
        results = {key: value[k-start].tolist() for key, value in stats.items()}
        results['words'] = words[k-start]
    return results


def get_co_cluster(data_name, model_name, state_name, n_clusters, layer=-1, top_k=100,
                   mode='positive', seed=0, method='cocluster'):
    """

    :param data_name:
    :param model_name:
    :param state_name:
    :param n_clusters:
    :param layer:
    :param top_k:
    :param mode: 'positive' or 'negative' or 'abs'
    :param seed: random seed
    :param method: 'cocluster' or 'bicluster'
    :return:
    """
    strength_list = get_empirical_strength(data_name, model_name, state_name, layer, top_k)
    strength_list = [strength_mat.reshape(-1) for strength_mat in strength_list]
    word_ids = []
    raw_data = []
    for i, strength_vec in enumerate(strength_list):
        if np.max(np.abs(strength_vec)) > 1e-8:
            word_ids.append(i)
            raw_data.append(strength_vec)

    raw_data = np.array(raw_data)
    if mode == 'positive':
        data = np.zeros(raw_data.shape, dtype=np.float32)
        data[raw_data >= 0] = raw_data[raw_data >= 0]
    elif mode == 'negative':
        data = np.zeros(raw_data.shape, dtype=np.float32)
        data[raw_data <= 0] = np.abs(raw_data[raw_data <= 0])
    elif mode == 'abs':
        data = np.abs(raw_data)
    elif mode == 'raw':
        data = raw_data
    else:
        raise ValueError("Unkown mode '{:s}'".format(mode))
    # print(data)
    n_jobs = 1  # parallel num
    random_state = seed
    if method == 'cocluster':
        row_labels, col_labels = spectral_co_cluster(data, n_clusters, n_jobs, random_state)
    elif method == 'bicluster':
        row_labels, col_labels = spectral_bi_cluster(data, n_clusters, n_jobs, random_state)
    else:
        raise ValueError('Unknown method type {:s}, should be cocluster or bicluster!'.format(method))
    return raw_data, row_labels, col_labels, word_ids


@lru_cache(maxsize=32)
def get_pos_statistics(data_name, model_name, top_k=500):
    top = 100 if top_k <= 100 else 500 if top_k <= 500 else 1000

    tmp_file = '-'.join([data_name, model_name, 'pos_ratio', str(top)]) + '.pkl'
    tmp_file = get_path(_tmp_dir, tmp_file)

    def cal_fn():
        word_ids, tags = load_words_and_state(data_name, model_name, 'pos', diff=False)
        ids_tags = sort_by_id(word_ids, tags)
        tags_counters = []
        for i, tags in enumerate(ids_tags):
            if tags is None:
                continue
            counter = Counter(tags)
            total = len(tags)
            for key, count in counter.items():
                counter[key] = count / total
            tags_counters.append({'id': i, 'ratio': counter})
        return tags_counters

    return maybe_calculate(tmp_file, cal_fn)[:top_k]



##############
# Functions that used in backends
##############


def maybe_calculate(filename, cal_fn, *args, **kwargs):
    """
    Check whether a cached .pkl file exists.
    If exists, directly load the file and return,
    Else, call the `cal_fn`, dump the results to .pkl file specified by `filename`, and return the results.
    :param filename: the name of the target cached file
    :param cal_fn: a function that maybe called with `*args` and `**kwargs` if no cached file is found.
    :return: the pickle dumped object, if cache file exists, else return the return value of cal_fn
    """
    if file_exists(filename):
        with open(filename, 'rb') as f:
            results = pickle.loads(f.read())
    else:
        results = cal_fn(*args, **kwargs)
        before_save(filename)
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
    return results


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


def cal_state_statistics(states):
    """
    Calculate states statistics with regards to a word
    :param states: a list of state_mat of size [layer_num, layer_size] (state changes of the same word)
    :return: a list of length layer_num, each element is a dict of statistics
    """
    layer_num = states[0].shape[0]
    percents = [50, 82]
    stats = []
    for layer in range(layer_num):
        state_list = [state[layer] for state in states]
        states_mat = np.vstack(state_list)
        mean = np.mean(states_mat, axis=0)
        idx = np.argsort(mean)
        lows = []
        highs = []
        for percent in percents:
            lows.append(np.percentile(states_mat, (100-percent)/2, axis=0))
            highs.append(np.percentile(states_mat, 50 + percent/2, axis=0))
        stats.append({'mean': mean,
                      'low1': lows[0],
                      'high1': highs[0],
                      'low2': lows[1],
                      'high2': highs[1],
                      'sort_idx': idx
                      })
    return stats


def cal_empirical_strength(id_states, strength_func):
    """

    :param id_states: a list, with each
    :param strength_func: np.mean, etc
    :return: a list of strength mat of shape [n_layer, state_size]
    """
    state_shape = id_states[0][0].shape

    def strenth_map(states):
        if states is None:
            return np.zeros(state_shape)
        states_mat = np.stack(states, axis=0)
        return strength_func(states_mat)

    strength_list = list(map(strenth_map, id_states))
    return strength_list


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


def spectral_co_cluster(data, n_clusters, para_jobs=1, random_state=None):
    from sklearn.cluster.bicluster import SpectralCoclustering
    model = SpectralCoclustering(n_clusters, random_state=random_state, n_jobs=para_jobs)
    model.fit(data)
    row_labels = model.row_labels_
    col_labels = model.column_labels_
    return row_labels, col_labels


def spectral_bi_cluster(data, n_clusters, para_jobs=1, random_state=None):
    from sklearn.cluster.bicluster import SpectralBiclustering
    assert len(n_clusters) == 2, "n_cluster should be a tuple or list that contains 2 integer!"
    model = SpectralBiclustering(n_clusters, random_state=random_state, n_jobs=para_jobs,
                                 method='bistochastic', n_best=20, n_components=40)
    model.fit(data)
    row_labels = model.row_labels_
    col_labels = model.column_labels_
    return row_labels, col_labels

##############
# Basic Functions used in this module
##############


def fetch_freq_words(data_name, k=100):
    id_to_word = get_dataset(data_name, ['id_to_word'])['id_to_word']
    return id_to_word[:k]


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


if __name__ == '__main__':

    data_name = 'ptb'
    model_name = 'GRU-PTB'
    state_name = 'state'
    ###
    # Scripts that run tsne on states and produce .json file for front-end rendering
    ###
    print('loading states...')
    #
    # sample = get_state_signature(data_name, model_name, state_name, [1], 5000, 50)/10
    #
    # solution = tsne_project(sample, 40.0, 50, 50)
    # labels = ([1] * (solution.shape[0])) # + ([0] * (solution.shape[0] // 2))
    # solution2json(solution, [0, 600], labels, get_path('_cached', 'gru-state-tsne.json'))
    # print("tsne saved")

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
    strength_mat = get_empirical_strength(data_name, model_name, state_name, layer=-1, top_k=50)
    id_to_word = get_dataset(data_name, ['id_to_word'])['id_to_word']
    word_list = id_to_word[:50]
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


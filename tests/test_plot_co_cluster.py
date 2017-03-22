import numpy as np
# import matplotlib.pyplot as plt

from rnnvis.state_processor import get_co_cluster
from rnnvis.plotter import matshow
from rnnvis.db import get_dataset
from rnnvis.utils.io_utils import get_path


if __name__ == '__main__':

    data_name = 'ptb'
    model_name = 'LSTM-PTB'
    state_name = 'state_c'
    # diff = False
    modes = ['abs', 'negative', 'positive', 'raw']
    mode = modes[3]

    # method = 'bicluster'
    # n_clusters = (20, 20)
    # n_str = '-'.join([str(n_cluster) for n_cluster in n_clusters])
    method = 'cocluster'
    n_clusters = 20
    n_str = str(n_clusters)
    top_k = 600

    results = get_co_cluster(data_name, model_name, state_name, n_clusters, -1, top_k, mode=mode, method=method)

    data, row_labels, col_labels = results[:3]
    id_to_word = get_dataset(data_name, ['id_to_word'])['id_to_word']
    word_list = id_to_word[:top_k]
    row_sort_idx = np.argsort(row_labels)
    col_sort_idx = np.argsort(col_labels)
    mat = data[row_sort_idx]
    mat = mat[:, col_sort_idx]
    print(mat.shape)
    matshow(mat, col_sort_idx, [word_list[i] for i in row_sort_idx],
            get_path('co-cluster' + '-' + method + '-' + mode + '-' + n_str + '.png'))

    # matshow(data, None, word_list, get_path('co-cluster' + '-' + 'raw' + '.png'))
    # plt.show()

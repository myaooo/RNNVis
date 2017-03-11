"""
The backend manager for handling the models
"""

import hashlib
import os

from rnnvis.datasets.data_utils import Feeder, SentenceProducer
from rnnvis.utils.io_utils import get_path, assert_path_exists
from rnnvis.procedures import build_model, pour_data
# from rnnvis.db import language_model
from rnnvis.rnn.eval_recorder import BufferRecorder, StateRecorder
from rnnvis.state_processor import get_state_signature, get_empirical_strength, strength2json, \
    get_tsne_projection, solution2json, get_co_cluster, get_state_statistics
from rnnvis.datasets.text_processor import tokenize

_config_dir = 'config'
_data_dir = 'cached_data'
_model_dir = 'models'


class ModelManager(object):

    _available_models = {
        'PTB-LSTM': {'config': 'lstm.yml'},
        # 'Shakespeare': {'config': 'shakespeare.yml'},
        'IMDB': {'config': 'imdb-tiny.yml'},
        'PTB-GRU': {'config': 'gru.yml'},
        'PTB-SMALL': {'config': 'lstm-small-1.yml'}
    }

    def __init__(self):
        self._models = {}
        self._train_configs = {}
        # self._records = {}
        # self._data = {}

    @property
    def available_models(self):
        return list(self._available_models.keys())

    def get_config_filename(self, name):
        """
        Get the config file path of a given model
        :param name: the name of the model, should be in _available_models
        :return: file path if the name is in _available_models, else None
        """
        if name in self._available_models:
            return get_path(_config_dir, self._available_models[name]['config'])
        else:
            return None

    def _get_model(self, name, train=False):
        if name in self._models:
            return self._models[name]
        else:
            flag = self._load_model(name, train)
            return self._models[name] if flag else None

    def _load_model(self, name, train=False):
        if name in self._available_models:
            config_file = get_path(_config_dir, self._available_models[name]['config'])
            model, train_config = build_model(config_file)
            model.add_generator()
            model.add_evaluator(1, 1, 100, True)
            if not train:
                # If not training, the model should already be trained
                assert_path_exists(get_path(_model_dir, model.name))
                model.restore()
            self._models[name] = model
            self._train_configs[name] = train_config
            return True
        else:
            print('WARN: Cannot find model with name {:s}'.format(name))
            return False

    def model_generate(self, name, seeds, max_branch=1, accum_cond_prob=0.9,
                       min_cond_prob=0.0, min_prob=0.0, max_step=10, neg_word_ids=None):
        """
        :param name: name of the model
        :param seeds: a list of word_id or a list of words
        :param max_branch: the maximum number of branches at each node
        :param accum_cond_prob: the maximum accumulate conditional probability of the following branches
        :param min_cond_prob: the minimum conditional probability of each branch
        :param min_prob: the minimum probability of a branch (note that this indicates a multiplication along the tree)
        :param max_step: the step to generate
        :param neg_word_ids: a set of neglected words' ids.
        :return:
        """
        model = self._get_model(name)
        if model is None:
            return None
        return model.generate(seeds, None, max_branch, accum_cond_prob, min_cond_prob, min_prob, max_step, neg_word_ids)

    def model_evaluate_sequence(self, name, sequences):
        """
        :param name:
        :param sequence: a str of sentence, or a list of words
        :return:
        """
        model = self._get_model(name)
        if model is None:
            return None
        if isinstance(sequences, str):
            tokenized_sequences = tokenize(sequences, eos=True, remove_punct=True)
            # flat_sequence = [items for sublist in tokenized_sequences for items in sublist]
            sequences = [model.get_id_from_word(sequence) for sequence in tokenized_sequences]
        config = self._train_configs[name]
        max_len = max([len(sequence) for sequence in sequences])
        # print(sequence)
        recorder = BufferRecorder(config.dataset, name, 500)
        # if not isinstance(sequences, Feeder):
        producer = SentenceProducer(sequences, 1, max_len, num_steps=1)
        inputs = producer.get_feeder()
        # print(inputs)
        model.evaluator.record_every = max_len
        # model.evaluate_and_record(inputs, None, recorder, verbose=False)
        # return list(recorder.evals())[0]
        try:
            model.run_with_context(model.evaluator.evaluate_and_record, inputs, None, recorder, verbose=False)
            # evals() retursn a sequence of eval_doc, records pairs, only needs the first one
            tokens, records = zip(*recorder.evals())
            print(tokens)
            tokens = [model.get_word_from_id(token) for token in tokens]
            print(tokens)
            return tokens, records
        # except ValueError:
        #     print("ERROR: Fail to evaluate given sequence! Sequence length too large!")
        #     return None
        except:
            print("ERROR: Fail to evaluate given sequence!")
            return None

    def model_record_default(self, name, dataset):
        """
        record default datasets
        :param name: model name
        :param dataset: 'train', 'valid', 'test'
        :return: True or False, None if model not exists
        """
        model = self._get_model(name)
        if model is None:
            return None
        config = self._train_configs[name]
        assert dataset in ['test', 'train', 'valid'], "dataset should be 'train', 'valid' or 'test'"
        recorder = StateRecorder(config.dataset, model.name, 500)
        producers = pour_data(config.dataset, [dataset], 1, 1, config.num_steps)
        inputs, targets, epoch_size = producers[0]
        model.evaluator.record_every = config.num_steps
        try:
            model.evaluate_and_record(inputs, None, recorder, verbose=False)
            return True
        except:
            print("ERROR: Fail to evaluate given sequence!")
            return False

    def model_sentences_to_ids(self, name, sentences):
        model = self._get_model(name)
        if model is None:
            return None
        ids = [model.get_id_from_word(sentence) for sentence in sentences]
        return ids

    def model_state_signature(self, name, state_name, layers, sample_size=1000):
        model = self._get_model(name)
        if model is None:
            return None
        config = self._train_configs[name]
        model_name = model.name
        data_name = config.dataset
        return get_state_signature(data_name, model_name, state_name, layers, sample_size, dim=None).tolist()

    def model_strength(self, name, state_name, layers, top_k=100):
        model = self._get_model(name)
        if model is None:
            return None
        config = self._train_configs[name]
        strength_mat = get_empirical_strength(config.dataset, model.name, state_name, layers, top_k)
        id_to_word = model.id_to_word
        word_list = id_to_word[:top_k]
        return strength2json(strength_mat, word_list)

    def model_state_projection(self, name, state_name, layer=-1, method='tsne'):
        model = self._get_model(name)
        if model is None:
            return None
        config = self._train_configs[name]
        layer_num = len(model.cell_list)
        if method == 'tsne':
            tsne_solution = get_tsne_projection(config.dataset, model.name, state_name, layer, 5000, 50, 40.0)
            labels = [layer_num - 1 if layer < 0 else layer] * tsne_solution.shape[0]
            states_num = [0] * layer_num
            states_num[layer] = tsne_solution.shape[0]
            return solution2json(tsne_solution, states_num, labels)
        else:
            return None

    def model_co_cluster(self, name, state_name, n_cluster=2, layer=-1, top_k=100, mode='positive', seed=0):
        model = self._get_model(name)
        if model is None:
            return None
        config = self._train_configs[name]
        layer_num = len(model.cell_list)
        strength_mat, row_cluster, col_cluster = get_co_cluster(config.dataset, model.name, state_name, n_cluster,
                                                                layer, top_k, mode, seed)
        return strength_mat.tolist(), row_cluster.tolist(), col_cluster.tolist()

    def model_vocab(self, name, top_k=None):
        model = self._get_model(name)
        if model is None:
            return None
        if top_k is None:
            return model.id_to_word
        return model.id_to_word[:top_k]

    def state_statistics(self, name, state_name, diff=True, layer=-1, top_k=500, k=None):
        model = self._get_model(name)
        if model is None:
            return None
        config = self._train_configs[name]
        if isinstance(k, str):
            k = model.get_id_from_word(k.lower())[0]
        stats = get_state_statistics(config.dataset, model.name, state_name, diff, layer, top_k, k)
        return stats


def hash_tag_str(text_list):
    """Use hashlib.md5 to tag a hash str of a list of text"""
    return hashlib.md5(" ".join(text_list).encode()).hexdigest()

if __name__ == '__main__':
    print(os.path.realpath(__file__))

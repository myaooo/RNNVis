"""
Utilities for processing data
"""

import math
import collections
import random

import tensorflow as tf
import numpy as np


class Feeder(object):

    def deque(self, transpose=None):
        """
        deque the next input
        :param transpose: if False, return is of shape [batch_size, num_steps(, feature_size)],
            if transpose is set to True, return is of shape [num_steps, batch_size(,feature_size)].
            note that this param is just for test and debug, it will not be used in training scripts
        :return:
        """
        raise NotImplementedError("this is the base class of Feeder!")

    def __call__(self):
        return self.deque()

    @property
    def shape(self):
        raise NotImplementedError("this is the base class of Feeder!")

    @property
    def full_data(self):
        raise NotImplementedError("this is the base class of Feeder!")

    @property
    def epoch_size(self):
        raise NotImplementedError("this is the base class of Feeder!")

    @property
    def need_refresh(self):
        """
        Whether the next dequeued data is a new sequence and need to refresh the state
        :return: True or False
        """
        raise NotImplementedError("this is the base class of Feeder!")


class ListFeeder(Feeder):
    def __init__(self, raw_list, batch_size, repeat=1, epoch_num=None):
        self.data = raw_list
        self.batch_size = batch_size
        self.max_epoch_num = math.inf if epoch_num is None else int(epoch_num)
        self._epoch_size = len(raw_list) // batch_size
        self.epoch_num = 0
        self.i = 0
        self.repeat = repeat
        self._shape = [batch_size]

    def deque(self, transpose=None):
        if transpose is not None:
            raise NotImplementedError("ListFeeder has no transpose option!")
        start = self.i // self.repeat * self.batch_size
        _data = self.data[start:start+self.batch_size]
        self.i += 1
        if self.i >= self.epoch_size:
            self.i = 0
            self.epoch_num += 1
        if self.epoch_num > self.max_epoch_num:
            raise ValueError("Exceeds maximum epoch num!")
        return _data

    @property
    def epoch_size(self):
        return self._epoch_size

    @property
    def shape(self):
        return self._shape

    @property
    def full_data(self):
        return self.data

    @property
    def need_refresh(self):
        raise NotImplementedError("Don't support this method for list")


class InputFeeder(Feeder):
    def __init__(self, batched_data, num_steps, offset=0, epoch_num=None, transpose=False):
        self.i = 0
        self.data = batched_data
        self.num_steps = num_steps
        self._epoch_size = (batched_data.shape[1] - offset) // num_steps
        assert self.epoch_size > 0
        self.offset = offset
        self.max_epoch_num = math.inf if epoch_num is None else int(epoch_num)
        self.epoch_num = 0
        self.transpose = transpose
        self.embedding = batched_data.ndim == 3
        _shape = [num_steps, batched_data.shape[0]] if transpose else [batched_data.shape[0], num_steps]
        if self.embedding:
            _shape.append(batched_data.shape[2])
        self._shape = _shape

    def deque(self, transpose=None):
        start = self.i * self.num_steps + self.offset
        _data = self.data[:, start:(start + self.num_steps)]
        transpose = self.transpose if transpose is None else transpose
        if transpose:
            _data = _data.transpose([1, 0, 2] if self.embedding else [1, 0])
        self.i += 1
        if self.i >= self.epoch_size:
            self.i = 0
            self.epoch_num += 1
        if self.epoch_num > self.max_epoch_num:
            raise ValueError("Exceeds maximum epoch num!")
        return _data

    @property
    def shape(self):
        return self._shape

    @property
    def epoch_size(self):
        return self._epoch_size

    @property
    def full_data(self):
        return self.data

    @property
    def need_refresh(self):
        return True if self.i == 0 else False


class InputProducer(object):
    """
    A convenient input data producer which maintain a "queue" inside it instead of using Tensorflow queue
    The purpose of using this is for the sake of simplicity,
    since there are plenty of restrictions on Tensorflow Queues
    """
    def __init__(self, raw_data, batch_size):
        """
        :param raw_data: a list of word_ids (int), or a list of embeddings
        :param batch_size: size of a batch
        """
        data_len = len(raw_data)
        self.batch_len = data_len // batch_size
        assert self.batch_len > 0
        self.batch_size = batch_size
        data = np.array(raw_data[:self.batch_len * self.batch_size])
        if data.ndim == 1:
            self.data = data.reshape(batch_size, self.batch_len)
        else:
            self.data = data.reshape([batch_size, self.batch_len, -1])

    def get_feeder(self, num_steps, offset=0, epoch_num=None, transpose=False):
        return InputFeeder(self.data, num_steps, offset, epoch_num, transpose)


class SentenceFeeder(Feeder):
    def __init__(self, data, batch_size, num_steps=None, offset=0, epoch_num=None, transpose=False):
        self.i = 0
        self.data = data
        self.max_length = data.shape[1]
        self.num_steps = self.max_length if num_steps is None else num_steps
        self._sentence_size = self.max_length // self.num_steps
        self.batch_size = batch_size
        self.max_epoch_num = math.inf if epoch_num is None else int(epoch_num)
        self.epoch_num = 0
        self.offset = offset
        self.transpose = transpose
        self._epoch_size = self.data.shape[0] // batch_size * self.sentence_size
        self.embedding = data.ndim == 3
        _shape = [self.num_steps, batch_size] if transpose else [batch_size, self.num_steps]
        if self.embedding:
            _shape.append(data.shape[2])
        self._shape = _shape

    def deque(self, transpose=None):
        start_1 = (self.i // self.sentence_size) * self.batch_size
        start_2 = (self.i % self.sentence_size) * self.num_steps
        _data = self.data[start_1:(start_1 + self.batch_size), start_2:(start_2 + self.num_steps)]
        transpose = self.transpose if transpose is None else transpose
        if transpose:
            _data = _data.transpose([1, 0, 2] if self.embedding else [1, 0])
        self.i += 1
        if self.i >= self.epoch_size:
            self.i = 0
            self.epoch_num += 1
        if self.epoch_num > self.max_epoch_num:
            raise ValueError("Exceeds maximum epoch num!")
        return _data

    @property
    def shape(self):
        return self._shape

    @property
    def epoch_size(self):
        return self._epoch_size

    @property
    def full_data(self):
        return self.data

    @property
    def need_refresh(self):
        return True

    @property
    def sentence_size(self):
        return self._sentence_size


class SentenceProducer(object):
    """
    A convenient input data producer which provide sentence level inputs,
        the num_steps are directly set by the max_length of the sentence
    """
    def __init__(self, raw_data, batch_size, max_length, num_steps=None):
        """
        :param raw_data: a list of lists, with each element as word_id (int), or a word_embedding (list or ndarray)
            each nested list represents a sentence
        For word_id input, we pad -1 at the end of short sequence,
        For word_embedding (array) input, we pad zero-arrays at the end of short sequence
        """
        self.batch_size = batch_size
        # if max_length is not None:  # trim off those too long
        raw_data = [data for data in raw_data if len(data) <= max_length]
        self.sentence_num = len(raw_data) // batch_size * batch_size
        raw_data = raw_data[:self.sentence_num]
        self.sentence_length = [len(l) for l in raw_data]
        self.max_length = max_length
        self.num_steps = self.max_length if num_steps is None else num_steps
        assert self.max_length % self.num_steps == 0, "the max_length should be complete times of num_steps"
        if isinstance(raw_data[0][0], int):
            self.embedding = False
            # Do －1 paddings if word_id
            data = np.zeros((self.sentence_num, self.max_length), dtype=int) - 1
            for i, l in enumerate(raw_data):
                length = min(self.sentence_length[i], self.max_length)
                data[i, :length] = np.array(l[:length])
        else:
            self.embedding = True
            # Do zero paddings if embedding
            data = np.zeros((self.sentence_num, self.max_length, len(raw_data[0][0])), dtype=float)
            for i, l in enumerate(raw_data):
                length = min(self.sentence_length[i], self.max_length)
                data[i, :length, :] = np.array(l[:length])
        self.data = data

    def get_feeder(self, offset=0, epoch_num=None, transpose=False):
        return SentenceFeeder(self.data, self.batch_size, self.num_steps, offset, epoch_num, transpose)


def split(data_list, fractions=None, shuffle=False):
    """
    Split the data set into various fractions
    :param data_list: a list of elems
    :param fractions: the fractions of different parts
    :param shuffle: shuffle or not
    :return: split data as a list
    """
    if fractions is None:
        fractions = [0.9, 0.05, 0.05]
    if shuffle:
        random.shuffle(data_list)
    assert sum(fractions) <= 1.0
    total_size = len(data_list)
    splitted = []
    start = 0.0
    for i in fractions:
        end = start + i
        splitted.append(data_list[int(start*total_size):int(end*total_size)])
        start = end
        # print(start)
    return splitted


def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").replace("\n", "<eos>").split()


def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id, words


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def load_data_as_ids(data_paths, word_to_id_path=None):
    """
    Load the data from a list of paths, the first file will be used to build the vocabulary.
    :param data_paths: a list of paths
    :param word_to_id_path: a word to ids csv file
    :return: a tuple of (data_list, word_to_id, id_to_word):
        a list of data, each as an id numpy.ndarray, corresponds to the data in each path in the data_paths,
        and a word_to_id dict, and a word list, with index as their ids
    """
    if word_to_id_path is not None:
        raise NotImplementedError("Currently not support separate word_to_id file")
    word_to_id, id_to_word = build_vocab(data_paths[0])
    data_list = []
    for path in data_paths:
        data_list.append(file_to_word_ids(path, word_to_id))
    return data_list, word_to_id, id_to_word


def get_lm_data_producer(data, batch_size, num_steps, transpose=False):
    # train_data = valid_data
    producer = InputProducer(data, batch_size)
    inputs = producer.get_feeder(num_steps, transpose=transpose)
    targets = producer.get_feeder(num_steps, offset=1, transpose=transpose)
    return inputs, targets, targets.epoch_size


def get_sp_data_producer(data, label, batch_size, max_length, num_steps=None, transpose=False):
    s_producer = SentenceProducer(data, batch_size, max_length, num_steps)
    inputs = s_producer.get_feeder(transpose=transpose)
    targets = ListFeeder(label[:s_producer.sentence_num], batch_size, inputs.sentence_size)
    return inputs, targets, targets.epoch_size

"""
Utilities for processing data
"""

import math
import collections
import tensorflow as tf
import numpy as np


class Feeder(object):
    def __init__(self, batched_data, num_steps, offset=0, epoch_num=None, transpose=False):
        self.i = 0
        self.data = batched_data
        self.num_steps = num_steps
        self.epoch_size = (batched_data.shape[1] - offset) // num_steps
        assert self.epoch_size > 0
        self.offset = offset
        self.max_epoch_num = math.inf if epoch_num is None else int(epoch_num)
        self.epoch_num = 0
        self.transpose = transpose

    def dequeue(self, transpose=False):
        start = self.i * self.num_steps + self.offset
        _data = self.data[:, start:(start + self.num_steps)]
        if transpose:
            _data = _data.T
        self.i += 1
        if self.i >= self.epoch_size:
            self.i = 0
            self.epoch_num += 1
        if self.epoch_num > self.max_epoch_num:
            raise ValueError("Exceeds maximum epoch num!")
        return _data

    def __call__(self):
        return self.dequeue(self.transpose)


class InputProducer(object):
    """
    A convenient input data producer which maintain a "queue" inside it instead of using Tensorflow queue
    The purpose of using this is for the sake of simplicity,
    since there are plenty of restrictions on Tensorflow Queues
    """
    def __init__(self, raw_data, batch_size):
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
        return Feeder(self.data, num_steps, offset, epoch_num, transpose)


def split(data_list, fractions=None, shuffle=False):
    if shuffle:
        raise NotImplementedError("No support of text data shuffling!")
    if fractions is None:
        fractions = [0.9, 0.05, 0.05]
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


def data_batcher(raw_data, batch_size):
    """
    Convert a list of raw data to a Tensor shaped [batch_len, batch_size (, feature_size)]
    Tail data will be ignored
    :param raw_data: a list of raw data with len T, each element shaped () or [feature_size]
    :param batch_size:
    :return:
    """
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    raw_data = tf.convert_to_tensor(raw_data[: batch_size * batch_len])
    shape = raw_data.get_shape()
    target_shape = [batch_len, batch_size] if len(shape) == 1 else [batch_len, batch_size, shape[1].value]
    data = tf.reshape(raw_data, target_shape)
    return data


# def data_feeder(batched_data, num_steps, shift=0, name=None):
#     """Iterate on the raw data. Borrowed from TensorFlow code
#
#     This returns Tensors that are drawn from raw_data.
#
#     Args:
#     raw_data: data with shape [batch_len, batch_size]
#     num_steps: int, the number of unrolls.
#     name: the name of this operation (optional).
#
#     Returns:
#     A Tensor shaped [num_steps, batch_size].
#
#     """
#     with tf.name_scope(name, "data_feeder", [batched_data, num_steps]):
#         batch_len = tf.shape(batched_data)[0]
#         batch_size = tf.shape(batched_data)[1]
#         epoch_size = (batch_len - 1) // num_steps
#         # assertion = tf.assert_positive(
#         #     epoch_size,
#         #     message="epoch_size == 0, decrease batch_size or num_steps")
#         # with tf.control_dependencies([assertion]):
#         #     epoch_size = tf.identity(epoch_size, name="epoch_size")
#
#         i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
#         x = tf.slice(batched_data, [i * num_steps + shift, 0], [num_steps, batch_size])
#         # y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
#         return x


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


def get_data_producer(data, batch_size, num_steps):
    # train_data = valid_data
    producer = InputProducer(data, batch_size)
    inputs = producer.get_feeder(num_steps, transpose=True)
    targets = producer.get_feeder(num_steps, offset=1, transpose=True)
    return inputs, targets, targets.epoch_size
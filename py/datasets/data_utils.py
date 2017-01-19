"""
Utilities for processing data
"""

import tensorflow as tf
import numpy as np
import math


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


def data_feeder(batched_data, num_steps, shift=0, name=None):
    """Iterate on the raw data. Borrowed from TensorFlow code

    This returns Tensors that are drawn from raw_data.

    Args:
    raw_data: data with shape [batch_len, batch_size]
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

    Returns:
    A Tensor shaped [num_steps, batch_size].

    """
    with tf.name_scope(name, "data_feeder", [batched_data, num_steps]):
        batch_len = tf.shape(batched_data)[0]
        batch_size = tf.shape(batched_data)[1]
        epoch_size = (batch_len - 1) // num_steps
        # assertion = tf.assert_positive(
        #     epoch_size,
        #     message="epoch_size == 0, decrease batch_size or num_steps")
        # with tf.control_dependencies([assertion]):
        #     epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.slice(batched_data, [i * num_steps + shift, 0], [num_steps, batch_size])
        # y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
        return x

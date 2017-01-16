"""
Utilities for processing data
"""

import tensorflow as tf


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


def data_feeder(raw_data, num_steps, shift=0, name=None):
    """Iterate on the raw data. Borrowed from TensorFlow code

    This returns Tensors that are drawn from raw_data.

    Args:
    raw_data: data with shape [batch_len, batch_size]
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

    Returns:
    A Tensor shaped [num_steps, batch_size].

    """
    with tf.name_scope(name, "data_feeder", [raw_data, num_steps]):
        batch_len = tf.shape(raw_data)[0]
        batch_size = tf.shape(raw_data)[1]
        epoch_size = (batch_len - 1) // num_steps
        # assertion = tf.assert_positive(
        #     epoch_size,
        #     message="epoch_size == 0, decrease batch_size or num_steps")
        # with tf.control_dependencies([assertion]):
        #     epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.slice(raw_data, [i * num_steps + shift, 0], [num_steps, batch_size])
        # y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
        return x

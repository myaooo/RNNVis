"""
Utilities for processing data
"""

import tensorflow as tf


def data_feeder(raw_data, num_steps, shift=0, name=None):
    """Iterate on the raw data. Borrowed from TensorFlow code

    This returns Tensors that are drawn from raw_data.

    Args:
    raw_data: data with shape [batch_size, batch_len]
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

    Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

    Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "data_feeder", [raw_data, num_steps]):
        batch_len = tf.shape(raw_data)[1]
        batch_size = tf.shape(raw_data)[0]
        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

            i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
            x = tf.slice(raw_data, [0, i * num_steps + shift], [batch_size, num_steps])
        # y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
        return x

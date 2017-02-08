"""
Helper functions for sentence (variable length) manipulation
"""

import tensorflow as tf


def sequence_length(sequence):
    """
    Get the length tensor of a batched_sequence
        when embedding, or say, input sequence is a 3D tensor, the empty part should be filled with 0.s
        whe word_id, or say, input sequence is a 2D tensor, the empty part should be filled with -1s
    :param sequence: a Tensor of shape [batch_size, max_length(, embedding_size)]
    :return: a 1D Tensor of shape (batch_size,) representing the length of the sequence
    """
    embedding = len(sequence.get_shape()) == 3
    if embedding:
        # zeros will be 0., others will be 1.
        used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
    else:
        # -1 will be 0, others will be 1.
        used = tf.sign(sequence+1)
    length = tf.reduce_sum(used, axis=1)
    length = tf.cast(length, tf.int32)
    return length


def flatten_var_length(sequences):
    """

    :param sequences: a 2D Tensor with shape [batch_size, max_length] of type tf.int32, with -1 masking
    :param length: the length tensor of shape [batch_size]
    :return:
    """
    used = tf.reshape(tf.sign(sequences + 1), [-1])
    flat = tf.reshape(sequences, [-1])
    return tf.boolean_mask(flat, used)


def last_relevant(output, length):
    """
    When dealing with variable length inputs, we will only be interested in the last relevant output
    :param output: a Tensor of shape [batch_size, max_length, feature_size]
    :param length: the length Tensor returned by sequence_length
    :return: the last outputs in the sequences, a 2D Tensor with shape [batch_size, feature_size]
    """
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


def cost(output, target):
    # Compute cross entropy for each frame.
    cross_entropy = target * tf.log(output)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    cross_entropy *= mask
    # Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(cross_entropy)


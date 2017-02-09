"""
Defines all losses related functions
"""

import numpy as np
import tensorflow as tf

from py.rnn.varlen_support import last_relevant


def softmax(x, axis=None):
    if axis is None:
        x_max = np.max(x)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x)
    if axis == 1:
        x = x.T
    x_max = np.max(x, axis=0)
    e_x = np.exp(x - x_max)
    sm = e_x / np.sum(e_x, axis=0)
    return sm if axis == 0 else sm.T


def sequence_loss(outputs, targets):
    """
    Weighted cross-entropy loss for a sequence of logits (per example).
    :param outputs: a 2D Tensor of shape [batch_size, output_size]
    :param targets: a 1D int32 Tensors of shape [batch_size]
    :return: a scalar tensor denoting the average loss of each example
    """
    if len(outputs.get_shape()) == 2:
        flatten_shape = tf.shape(targets)
    elif len(outputs.get_shape()) == 3:
        shape = outputs.get_shape().as_list()
        outputs = tf.reshape(outputs, [-1, shape[2]])
        targets = tf.reshape(targets, [-1])
        flatten_shape = tf.shape(targets)
    else:
        raise ValueError("outputs must be 2D or 3D tensor!")
    _loss = tf.nn.seq2seq.sequence_loss([outputs], [targets], [tf.ones(flatten_shape, dtype=outputs.dtype)])
    return _loss


def sentence_loss(last_outputs, targets):
    return sequence_loss(last_outputs, targets)

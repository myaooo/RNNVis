"""
Trainer handling TensorFlow Graph training computations
"""

import tensorflow as tf
from . import rnn


class Trainer(object):
    """
    Trainer Class
    """
    def __init__(self, model, optimizer, learning_rate):
        if not isinstance(model, rnn.RNN):
            raise TypeError("model should be of class RNN!")
        self.model = model
        self.optimizer = optimizer
        # self.initializer = initializer
        self._lr = learning_rate
        self.train_input_holder = tf.placeholder(model.input_dtype, model.input_shape, name='input')
        self.train_target_holder = tf.placeholder(model.output_dtype, model.output_shape, name='target')

    def train(self, input_, target, epoch_num, batch_size,
              validation_set=None, validation_batch_size=None, input_is_queue=False):
        """
        Training using given input and target data
        :param input_: should be a list of input sequence, each element of the list is a input sequence specified by x.
        :param target: should be of size [num_seq, seq_length, ...]
        :param epoch_num: number of training epochs
        :param batch_size: batch_size
        :param validation_set: Validation set, should be (input, output)
        :param validation_batch_size: batch_size of validation set
        :return: None
        """
        # cell weights are already initialized upon creation
        with tf.variable_scope(self.model.name, reuse=True):
            num_seq = tf.shape(input_)[0]
            num_it = num_seq // batch_size

            i = tf.train.range_input_producer(num_it).deque()



"""
RNN Model Classes
"""

import tensorflow as tf

BasicRNNCell = tf.nn.rnn_cell.BasicRNNCell
BasicLSTMCell = tf.nn.rnn_cell.BasicLSTMCell
LSTMCell = tf.nn.rnn_cell.LSTMCell
GRUCell = tf.nn.rnn_cell.GRUCell


class RNN(object):
    """A class of RNN Model that wraps TF RNNCells"""
    def __init__(self, cell=None, supervisor=None):
        """
        :param cell: a instance of RNN cell
        :param supervisor: a tf.train.Supervisor instance that helps manage the summary ops
        :return: a empty RNN model
        """
        self._cell = cell
        self.cell_list = []
        self.supervisor = supervisor
        self.loss = None
        self._train_op = None

    def add_cell(self, cell):
        """
        push back a cell
        :param cell: a RNN cell, should be instance of tf.nn.rnn_cell.RNNCell
        :return:
        """
        self.cell_list.append(cell)

    def compile(self):
        """
        Compile the model.
        Should be called before training or running the model
        :return: None
        """
        if self._cell is not None:
            self._cell = tf.nn.rnn_cell.MultiRNNCell(cells=self.cell_list)

    def infer(self, incoming):
        pass

    def train(self, epoch_num):
        pass

    def one_step(self, sess, _input, _output):
        pass

    @property
    def cell(self):
        return self._cell

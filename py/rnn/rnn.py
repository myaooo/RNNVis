"""
RNN Model Classes
"""

import tensorflow as tf
from .trainer import Trainer
from .config import *

BasicRNNCell = tf.nn.rnn_cell.BasicRNNCell
BasicLSTMCell = tf.nn.rnn_cell.BasicLSTMCell
LSTMCell = tf.nn.rnn_cell.LSTMCell
GRUCell = tf.nn.rnn_cell.GRUCell
DropOutWrapper = tf.nn.rnn_cell.DropoutWrapper
EmbeddingWrapper = tf.nn.rnn_cell.EmbeddingWrapper
InputProjectionWrapper = tf.nn.rnn_cell.InputProjectionWrapper
OutputProjectionWrapper = tf.nn.rnn_cell.OutputProjectionWrapper

int32 = tf.int32


def _get_optimizer(optimizer):
    """
    Simple helper to get TensorFlow Optimizer
    :param optimizer: a str specifying the name of the optimizer to use,
        or a callable function that is already a TF Optimizer
    :return: a TensorFlow Optimizer
    """
    if callable(optimizer):
        return optimizer


def loss_by_example(outputs, targets):
    """
    Weighted cross-entropy loss for a sequence of logits (per example).
    :param outputs: a list of 2D Tensor of shape [batch_size, output_size]
    :param targets: List of 1D batch-sized int32 Tensors of the same length as output
    :return: a scalar tensor denoting the average loss of each example
    """
    batch_size = tf.shape(targets[0])[0]
    _loss = tf.nn.seq2seq.sequence_loss_by_example(outputs, targets, [tf.ones(tf.shape(targets[0]))]*len(outputs))
    return _loss / batch_size

# def create_cell_under_scope(cell, name, **kwargs):
#     """
#     A helper function to use for create RNN cells under specific TensorFlow VariableScope
#     :param cell: should be subclasses of tf.nn.rnn_cell.BasicRNNCell
#     :param name: the name or scope used for managing the cell
#     :param kwargs: the arguments used to create the cell
#     :return: a cell managed under scope
#     """
#     with tf.variable_scope(name, reuse=None):
#         return cell(**kwargs)


class Config(object):
    """
    Helper Class to create a RNN model
    """
    def __init__(self,
                 cell=BasicRNNCell,
                 layer_num=3,
                 layer_size=256,
                 num_step=10,
                 batch_size=30,
                 initializer=tf.random_uniform_initializer(-0.1, 0.1)):
        self.cell = cell
        self.layer_num = layer_num
        self.layer_size = layer_size
        self.num_step = num_step
        self.batch_size = batch_size
        self.initializer = initializer
        # self.init_scale = init_scale


class RNNModel(object):
    """
    A RNN Model unrolled from a RNN instance
    """
    def __init__(self, rnn, batch_size, num_steps, name=None):
        assert isinstance(rnn, RNN)
        self._rnn = rnn
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.name = name or "UnRolled"
        # The abstract name scope is not that easily dealt with, try to make the transparent to user
        with tf.name_scope(name):
            with tf.variable_scope(self._rnn.name, reuse=True, initializer=self._rnn.initializer):
                # Define computation graph
                # self._init_state = rnn.cell.zero_state(batch_size, data_type())ß
                self.state = rnn.cell.zero_state(batch_size, data_type())
                self.input_holders = [tf.placeholder(self._rnn.input_dtype, self._rnn.input_shape)] * num_steps
                self.output, self.final_state = tf.nn.rnn(self._rnn.cell, self.input_holders, self.state)
                self.target_holders = [tf.placeholder(self._rnn.output_dtype, self._rnn.output_shape)] * num_steps
                self.loss = self._rnn.loss_func(self.output, self.target_holders)

    def feed_state(self, state):
        """
        Create feed_dict specifying the state of rnn input state
        :param state: a tuple of states which can be convert to numpy arrays
        :return: a feed_dict used in sess.run()
        """
        # state_size = self._rnn.cell.state_size
        feed_dict = {}
        # if tf.nest.is_sequence(state_size): For multiRNNCell, the state must be a tuple
        # state is tuple
        for i, s in self.state:
            # s is tuple
            if isinstance(s, tf.nn.rnn_cell.LSTMStateTuple):
                feed_dict[s.c] = state[i].c
                feed_dict[s.h] = state[i].h
            else:
                feed_dict[s] = state[i]
        return feed_dict

    def init_state(self, sess):
        return sess.run(self.state)


class RNN(object):
    """
    A helper class that wraps TF RNNCells
    This class is used for wrapping RNN Model definition. You cannot use this class directly for computation.
    For computation (training, evaluating), use RNN.unroll() to create RNNModel,
    which create TF computation Graph for computation.
    """
    def __init__(self, name="RNN", initializer=None):
        """
        :param name: a str, used to create variable scope
        :return: a empty RNN model
        """
        self.name = name
        self.initializer = initializer if initializer is not None else tf.random_uniform_initializer(-0.1,0.1)
        self.input_shape = None
        self.input_dtype = None
        self.output_shape = None
        self.output_dtype = None

        self._cell = None
        self.cell_list = []
        self.trainer = None
        self.loss_func = None
        self.is_compiled = False
        self.map_to_embedding = None
        self.embedding_size = None
        self.vocab_size = None

    def set_input(self, dshape, dtype, vocab_size, embedding_size=None):
        """
        set the input data shape of the model
        :param dshape: tuple or a list, [batch_size, ...]
            typically [None] for word-id input,
            for word embeddings, should be [None, feature_dim]
        :param dtype: tensorflow data type
        :return: None
        """
        self.input_shape = list(dshape)
        self.input_dtype = dtype
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

    def set_output(self, dshape, dtype):
        """
        Set the target data shape and type
        :param dshape: tuple or a list, [batch_size, ...],
            typically, for language modeling problem, the shape should match the input shape
        :param dtype: tensorflow data type
        :return: None
        """
        self.output_shape = list(dshape)
        self.output_dtype = dtype

    def set_loss_func(self, loss_func):
        """
        Set the loss function of the model
        :param loss_func: a function of form loss = loss_func(outputs, targets), where loss is a scalar Tensor
        :return: None
        """
        self.loss_func = loss_func

    def _add_cell(self, cell, *args, **kwargs):
        """
        A helper function to use for create RNN cells under specific TensorFlow VariableScope
        :param cell: should be subclasses of tf.nn.rnn_cell.BasicRNNCell
        :param name: the name or scope used for managing the cell
        :param kwargs: the arguments used to create the cell
        :return: a cell managed under scope
        """
        with tf.variable_scope(self.name, reuse=None, initializer=self.initializer):
            self.cell_list.append(cell(*args, **kwargs))

    def compile(self):
        """
        Compile the model. Should be called before training or running the model.
        Basically, this function concat all the cells together, and do checkings on model configurations
        :return: None
        """
        if self.is_compiled:  # In case of multiple compiles
            print("Already compiled!")
            return
        if self.input_shape is None or self.input_dtype is None:
            raise ValueError("input_shape or input_dtype is None, call set_input first!")
        if self.output_shape is None or self.output_dtype is None:
            raise ValueError("output_shape or output_dtype is None, call set_output first!")
        # with tf.variable_scope(self.name, reuse=None, initializer=config.initializer) as scope:
        # if self._cell is not None:
            # This operation creates no tf.Variables, no need for using variable scope
        # Wrappers are not efficient, here for use them for convenience
        if self.embedding_size:
            self.cell_list[0] = EmbeddingWrapper(self.cell_list[0], self.vocab_size, self.embedding_size)
        if self.cell_list[-1].output_size != self.output_shape[-1]:
            # if the last cell's output_size does not match intended output shape, we need a projection
            self.cell_list[-1] = OutputProjectionWrapper(self.cell_list[-1], self.output_shape[-1])
        self._cell = tf.nn.rnn_cell.MultiRNNCell(cells=self.cell_list)
        self.is_compiled = True

    def unroll(self, batch_size, num_steps, name=None):
        """
        Unroll the RNN Cell into a RNN Model with fixed num_steps and batch_size
        :param batch_size: batch_size of the model
        :param num_steps: unrolled num_steps of the model
        :param name: name of the model
        :return:
        """
        return RNNModel(self, batch_size, num_steps, name)

    def train(self, input_, target, num_steps, epoch_num, batch_size, optimizer, learning_rate=0.001,
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
        if self.trainer is None:
            model = self.unroll(batch_size, num_steps, 'Train')
            self.trainer = Trainer(model, optimizer, learning_rate)
        else:
            self.trainer.optimizer = optimizer
            self.trainer._lr = learning_rate

        self.trainer.train(input_, target, num_steps, epoch_num, batch_size)

    @property
    def cell(self):
        return self._cell

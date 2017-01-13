"""
RNN Model Classes
"""

import tensorflow as tf
from .trainer import Trainer

BasicRNNCell = tf.nn.rnn_cell.BasicRNNCell
BasicLSTMCell = tf.nn.rnn_cell.BasicLSTMCell
LSTMCell = tf.nn.rnn_cell.LSTMCell
GRUCell = tf.nn.rnn_cell.GRUCell

int32 = tf.int32

# Avoid too complicated design

# class DNN(object):
#     """Base class of DNN model"""
#     def __init__(self, name):
#         self.name = name
#         self.input_shape = None
#         self.input_dtype = None
#         self.output_shape = None
#         self.output_dtype = None
#         self.input_holder = None
#         self.target_holder = None
#         self.train_input_holder = None
#         self.train_target_holder = None
#         self._train_ops = {}
#         self._eval_ops = {}
#
#     def set_input(self, dshape, dtype):
#         """
#         set the input data shape of the model
#         :param dshape: tuple or a list, [batch_size, ...]
#             typically [None, length] for word-id input,
#             for word embeddings, should be [None, length, embedding_len]
#         :param dtype: tensorflow data type
#         :return:
#         """
#         self.input_shape = list(dshape)
#         self.input_dtype = dtype
#
#     def set_output(self, dshape, dtype):
#         """
#         Set the target data shape and type
#         :param dshape: tuple or a list, [batch_size, ...],
#             typically, for language modeling problem, the shape should match the input shape
#         :param dtype: tensorflow data type
#         :return:
#         """
#         self.output_shape = list(dshape)
#         self.output_dtype = dtype
#
#     def compile(self):
#         """
#         Do all the checking and initializations before building computation graphs
#         :return: None
#         """
#         if self.input_shape is None:
#             raise ValueError("input_shape is None, call set_input first!")


def _get_optimizer(optimizer):
    """
    Simple helper to get TensorFlow Optimizer
    :param optimizer: a str specifying the name of the optimizer to use,
        or a callable function that is already a TF Optimizer
    :return: a TensorFlow Optimizer
    """
    if callable(optimizer):
        return optimizer


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
                 initializer=tf.random_uniform_initializer(-0.1,0.1)):
        self.cell = cell
        self.layer_num = layer_num
        self.layer_size = layer_size
        self.num_step = num_step
        self.batch_size = batch_size
        self.initializer = initializer
        # self.init_scale = init_scale


class RNN(object):
    """A class of RNN Model that wraps TF RNNCells"""
    def __init__(self, name="RNN", config=None):
        """
        :param name: a str, used to create variable scope
        :param cell: a instance of RNN cell, if None, then need to explicitly add them one by one using add_cell
        :return: a empty RNN model
        """
        self.name = name
        self.config = config if config is not None else Config()
        self.input_shape = None
        self.input_dtype = None
        self.output_shape = None
        self.output_dtype = None
        # Place Holders
        self.input_holder = None
        self.target_holder = None
        self.train_input_holder = None
        self.train_target_holder = None

        self._train_ops = {}
        self._eval_ops = {}
        self._step_ops = {}
        self._cell = None
        self.cell_list = []
        self.trainer = None
        self.loss = None
        self.is_compiled = False
        self._state = None  # Current state
        self.output_holder = None
        self.scope = tf.variable_scope(self.name)
        # self.sess = tf.Session() if supervisor is None else supervisor.managed_session()

    def set_input(self, dshape, dtype):
        """
        set the input data shape of the model
        :param dshape: tuple or a list, [batch_size, ...]
            typically [None, length] for word-id input,
            for word embeddings, should be [None, length, embedding_len]
        :param dtype: tensorflow data type
        :return:
        """
        self.input_shape = list(dshape)
        self.input_dtype = dtype

    def set_output(self, dshape, dtype):
        """
        Set the target data shape and type
        :param dshape: tuple or a list, [batch_size, ...],
            typically, for language modeling problem, the shape should match the input shape
        :param dtype: tensorflow data type
        :return:
        """
        self.output_shape = list(dshape)
        self.output_dtype = dtype

    def _add_cell(self, cell, *args, **kwargs):
        """
        A helper function to use for create RNN cells under specific TensorFlow VariableScope
        :param cell: should be subclasses of tf.nn.rnn_cell.BasicRNNCell
        :param name: the name or scope used for managing the cell
        :param kwargs: the arguments used to create the cell
        :return: a cell managed under scope
        """
        with tf.variable_scope(self.name, reuse=None, initializer=self.config.initializer):
            self.cell_list.append(cell(*args, **kwargs))

    # def model(self, inputs, initial_state=None, sequence_length=None, scope_name=None):
    #     return tf.nn.rnn(self._cell, inputs, initial_state, sequence_length, scope_name)

    def compile(self):
        """
        Compile the model.
        Should be called before training or running the model
        :return: None
        """
        if self.is_compiled:  # In case of multiple compiles
            print("Already compiled!")
            return
        if self.input_shape is None or self.input_dtype is None:
            raise ValueError("input_shape or input_dtype is None, call set_input first!")
        if self.output_shape is None or self.output_dtype is None:
            raise ValueError("output_shape or output_dtype is None, call set_output first!")
        config = self.config
        with tf.variable_scope(self.name, reuse=None, initializer=config.initializer) as scope:
            if self._cell is not None:
                self._cell = tf.nn.rnn_cell.MultiRNNCell(cells=self.cell_list)

            # # Build placeholders
            # with tf.name_scope('train'):
            #     self.train_input_holder = tf.placeholder(self.input_dtype, self.input_shape, name='input')
            #     self.train_target_holder = tf.placeholder(self.output_dtype, self.output_shape, name='target')
            #     outputs = tf.nn.dynamic_rnn(self._cell, self.train_input_holder)
            with tf.name_scope('eval'):
                self.input_holder = tf.placeholder(self.input_dtype, self.input_shape, name='input')
                self.target_holder = tf.placeholder(self.output_shape, self.output_shape, name='target')
                self._state = self._cell.zero_state(config.batch_size, tf.float32)
                self.output_holder = tf.nn.rnn(self._cell, inputs=self.input_holder, initial_state=self._state)
            # outputs = self.model(self.train_input_holder)


    # def step(self, incoming, initial_state, train = False):
    #     """
    #     Step
    #     :param incoming:
    #     :param initial_state:
    #     :return:
    #     """
    #     tf.nn.dynamic_rnn(self._cell, incoming, initial_state)

    def train(self, input_, target, epoch_num, batch_size, optimizer, learning_rate=0.001,
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
            self.trainer = Trainer(self, optimizer, learning_rate)
        else:
            self.trainer.optimizer = optimizer

        self.trainer.train(input_, target, epoch_num, batch_size, validation_set, validation_batch_size, input_is_queue)

    def one_step(self, sess, _input, _output):
        pass

    @property
    def cell(self):
        return self._cell

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
MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell
DropOutWrapper = tf.nn.rnn_cell.DropoutWrapper
# EmbeddingWrapper = tf.nn.rnn_cell.EmbeddingWrapper
InputProjectionWrapper = tf.nn.rnn_cell.InputProjectionWrapper
OutputProjectionWrapper = tf.nn.rnn_cell.OutputProjectionWrapper

int32 = tf.int32


def loss_by_example(outputs, targets):
    """
    Weighted cross-entropy loss for a sequence of logits (per example).
    :param outputs: a 3D Tensor of shape [num_steps, batch_size, output_size]
    :param targets: List of 2D int32 Tensors of shape [num_steps, batch_size]
    :return: a scalar tensor denoting the average loss of each example
    """
    shape = tf.shape(outputs)
    flatten_size = shape[0] * shape[1]
    outputs = tf.reshape(outputs, [-1, shape[2]])
    targets = tf.reshape(targets, [-1])
    _loss = tf.nn.seq2seq.sequence_loss([outputs], [targets], [tf.ones([flatten_size], dtype=data_type())])
    return _loss


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
    def __init__(self, rnn, batch_size, num_steps, keep_prob=None, name=None):
        assert isinstance(rnn, RNN)
        self._rnn = rnn
        self._cell = rnn.cell
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.name = name or "UnRolled"
        if keep_prob is not None:
            cell_list = [DropOutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob) for cell in rnn.cell_list]
            self._cell = MultiRNNCell(cell_list, state_is_tuple=True)
        # The abstract name scope is not that easily dealt with, try to make the transparent to user
        with tf.name_scope(name):
            reuse = None if len(self._rnn.models) == 0 else True

            with tf.variable_scope(rnn.name, reuse=reuse, initializer=rnn.initializer):
                # Define computation graph
                input_shape = list(rnn.input_shape)
                target_shape = list(rnn.target_shape)
                input_shape[0] = target_shape[0] = batch_size
                self.state = rnn.cell.zero_state(batch_size, data_type())
                self.input_holders = tf.placeholder(rnn.input_dtype, [num_steps] + input_shape)
                # ugly hacking for EmbeddingWrapper Badness
                if rnn.map_to_embedding:
                    inputs = rnn.map_to_embedding(self.input_holders)
                else:
                    inputs = self.input_holders
                self.output, self.final_state = \
                    tf.nn.dynamic_rnn(self.cell, inputs, initial_state=self.state, time_major=True)
                self.target_holders = tf.placeholder(rnn.target_dtype, [num_steps] + target_shape)
                self.loss = rnn.loss_func(self.output, self.target_holders)

                # self._assign_inputs = tf.assign(self.input_holders, )

    @property
    def cell(self):
        return self._cell

    @property
    def rnn(self):
        return self._rnn

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
        for i, s in enumerate(self.state):
            # s is tuple
            if isinstance(s, tf.nn.rnn_cell.LSTMStateTuple):
                feed_dict[s.c] = state[i].c
                feed_dict[s.h] = state[i].h
            else:
                feed_dict[s] = state[i]
        return feed_dict

    def feed_data(self, data, inputs=True):
        holders = self.input_holders if inputs else self.target_holders
        return {holders: data}

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
        self.target_shape = None
        self.target_dtype = None

        self._cell = None
        self.cell_list = []
        self.trainer = None
        self.loss_func = None
        self.is_compiled = False
        self.map_to_embedding = None
        self.embedding_size = None
        self.vocab_size = None
        self.supervisor = None
        self.models = []

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

    def set_output(self, dshape, dtype=tf.float32):
        """
        Set the target data shape and type
        :param dshape: tuple or a list, [batch_size, ...],
            typically, for language modeling problem, the shape should match the input shape
        :param dtype: tensorflow data type
        :return: None
        """
        self.output_shape = list(dshape)
        self.output_dtype = dtype

    def set_target(self, dshape, dtype=tf.int32):
        self.target_shape = dshape
        self.target_dtype = dtype

    def set_loss_func(self, loss_func):
        """
        Set the loss function of the model
        :param loss_func: a function of form loss = loss_func(outputs, targets), where loss is a scalar Tensor
        :return: None
        """
        self.loss_func = loss_func

    def add_cell(self, cell, *args, **kwargs):
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
        if self.loss_func is None:
            raise ValueError("loss_func is None, call set_loss_func first!")
        # with tf.variable_scope(self.name, reuse=None, initializer=config.initializer) as scope:
        # if self._cell is not None:
            # This operation creates no tf.Variables, no need for using variable scope
        # Wrappers are not efficient, here for use them for convenience

        if self.embedding_size:
            # EmbeddingWrapper does not work for tf.nn.rnn now
            # self.cell_list[0] = EmbeddingWrapper(self.cell_list[0], self.vocab_size, self.embedding_size)
            with tf.variable_scope(self.name, reuse=None, initializer=self.initializer):
                embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=data_type())
                self.map_to_embedding = lambda inputs: tf.nn.embedding_lookup(embedding, inputs)

        if self.cell_list[-1].output_size != self.output_shape[-1]:
            # if the last cell's output_size does not match intended output shape, we need a projection
            self.cell_list[-1] = OutputProjectionWrapper(self.cell_list[-1], self.output_shape[-1])
        self._cell = tf.nn.rnn_cell.MultiRNNCell(cells=self.cell_list)
        self.is_compiled = True

    def unroll(self, batch_size, num_steps, keep_prob=None, name=None):
        """
        Unroll the RNN Cell into a RNN Model with fixed num_steps and batch_size
        :param batch_size: batch_size of the model
        :param num_steps: unrolled num_steps of the model
        :param name: name of the model
        :return:
        """
        assert self.is_compiled
        self.models.append(RNNModel(self, batch_size, num_steps, keep_prob=keep_prob, name=name))
        return self.models[-1]

    def train(self, inputs, targets, num_steps, epoch_size, epoch_num, batch_size,
              optimizer, learning_rate=0.001, keep_prob=None, clipper=None, decay=None,
              valid_inputs=None, valid_targets=None, valid_batch_size=None, logdir=None):
        """
        Training using given input and target data
        TODO: Clean up this messy function
        :param inputs: should be a list of input sequence, each element of the list is a input sequence specified by x.
        :param targets: should be of size [num_seq, seq_length, ...]
        :param epoch_num: number of training epochs
        :param batch_size: batch_size
        :param validation_set: Validation set, should be (input, output)
        :param validation_batch_size: batch_size of validation set
        :return: None
        """
        assert self.is_compiled
        if self.trainer is None:
            model = self.unroll(batch_size, num_steps, keep_prob, name='Train')
            self.trainer = Trainer(model, optimizer, learning_rate, clipper, decay)
        else:
            self.trainer.optimizer = optimizer
            self.trainer._lr = learning_rate
        if valid_inputs is not None:
            self.trainer.valid_model = self.unroll(valid_batch_size, num_steps, name='Valid')
        print("Start Running Train Graph")
        self.trainer.train(inputs, targets, epoch_size, epoch_num, valid_inputs, valid_targets, save_path=logdir)

    @property
    def cell(self):
        return self._cell

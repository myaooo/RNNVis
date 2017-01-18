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
    _loss = tf.nn.seq2seq.sequence_loss_by_example([outputs], [targets], [tf.ones(flatten_shape, dtype=data_type())])
    return tf.reduce_mean(_loss)


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
        # Ugly hackings for DropoutWrapper
        if keep_prob is not None:
            cell_list = [DropOutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob) for cell in rnn.cell_list]
            self._cell = MultiRNNCell(cell_list, state_is_tuple=True)
        # The abstract name scope is not that easily dealt with, try to make the transparent to user
        with tf.name_scope(name):
            reuse = rnn.need_reuse
            with tf.variable_scope(rnn.name, reuse=reuse, initializer=rnn.initializer):
                # Build TF computation Graph
                input_shape = [batch_size] + list(rnn.input_shape)[1:]
                target_shape = [batch_size] + list(rnn.target_shape)[1:]
                self.state = self.cell.zero_state(batch_size, data_type())
                self.input_holders = tf.placeholder(rnn.input_dtype, [num_steps] + input_shape)
                self.target_holders = tf.placeholder(rnn.target_dtype, [num_steps] + target_shape)
                # ugly hacking for EmbeddingWrapper Badness
                inputs = self.input_holders if not rnn.map_to_embedding else rnn.map_to_embedding(self.input_holders)
                # Call TF api to create recurrent neural network
                self.outputs, self.final_state = \
                    tf.nn.dynamic_rnn(self.cell, inputs, initial_state=self.state, time_major=True)
                if rnn.project_output:
                    # rnn has output project, do manual projection for speed
                    outputs = tf.reshape(self.outputs, [-1, self.outputs.get_shape().as_list()[2]])
                    outputs = rnn.project_output(outputs)
                    targets = tf.reshape(self.target_holders, [-1])
                    self.loss = rnn.loss_func(outputs, targets)
                else:
                    self.loss = rnn.loss_func(self.outputs, self.target_holders)
        # Append self to rnn's model list
        rnn.models.append(self)

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
        self._map_to_embedding = None
        self._projcet_output = None
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
        self.cell_list.append(cell(*args, **kwargs))

    def compile(self):
        """
        Compile the model. Should be called before training or running the model.
        Basically, this function just do checkings on model configurations, it creates no tf.Variables or ops,
            it just do all the prepares before building the computation graph
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
        # This operation creates no tf.Variables, no need for using variable scope
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
        return RNNModel(self, batch_size, num_steps, keep_prob=keep_prob, name=name)

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
            self.trainer = Trainer(self,batch_size,num_steps,keep_prob, optimizer, learning_rate, clipper, decay)
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

    @property
    def need_reuse(self):
        return None if len(self.models) == 0 else True

    def map_to_embedding(self, inputs):
        if self.embedding_size:
            # The Variables are already created in the compile(), need to
            with tf.variable_scope('embedding', initializer=self.initializer):
                embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=data_type())
                return tf.nn.embedding_lookup(embedding, inputs)
        else:
            return None

    def project_output(self, outputs):
        if self.cell_list[-1].output_size != self.output_shape[-1]:
            with tf.variable_scope('project', initializer=self.initializer):
                project_w = tf.get_variable(
                    "project_w", [self.cell_list[-1].output_size, self.vocab_size], dtype=data_type())
                projcet_b = tf.get_variable("project_b", [self.vocab_size], dtype=data_type())
                return tf.matmul(outputs, project_w) + projcet_b
        else:
            return None

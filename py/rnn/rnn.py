"""
RNN Model Classes
"""

import logging
import math
import time
import os

import tensorflow as tf
import numpy as np
from py.datasets.data_utils import Feeder
from .command_utils import data_type, config_proto
from .evaluator import Evaluator
from .trainer import Trainer
from .generator import Generator

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
    _loss = tf.nn.seq2seq.sequence_loss([outputs], [targets], [tf.ones(flatten_shape, dtype=data_type())])
    return _loss


class RNNModel(object):
    """
    A RNN Model unrolled from a RNN instance
    """
    def __init__(self, rnn, batch_size, num_steps, keep_prob=None, name=None):
        assert isinstance(rnn, RNN)
        # if batch_size is None:
        #     batch_size = tf.Variable(20, dtype=tf.int32, trainable=False)
        #     self._new_batch_size = tf.placeholder(tf.int32, shape=())
        #     self._update_batch_size = tf.assign(batch_size, self._new_batch_size, name="update_batch_size")
        self._rnn = rnn
        self._cell = rnn.cell
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.name = name or "UnRolled"
        self.current_state = None
        # Ugly hackings for DropoutWrapper
        if keep_prob is not None:
            cell_list = [DropOutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob) for cell in rnn.cell_list]
            self._cell = MultiRNNCell(cell_list, state_is_tuple=True)
        # The abstract name scope is not that easily dealt with, try to make the transparent to user
        with tf.name_scope(name):
            reuse = rnn.need_reuse
            with tf.variable_scope(rnn.name, reuse=reuse, initializer=rnn.initializer):
                # Build TF computation Graph
                input_shape = [None] + list(rnn.input_shape)[1:]
                target_shape = [None] + list(rnn.target_shape)[1:]
                self.state = self.cell.zero_state(batch_size, rnn.output_dtype)
                self.input_holders = tf.placeholder(rnn.input_dtype, [num_steps] + input_shape, "input_holders")
                self.target_holders = tf.placeholder(rnn.target_dtype, [num_steps] + target_shape, "target_holders")
                # ugly hacking for EmbeddingWrapper Badness
                self.inputs = self.input_holders if not rnn.has_embedding else rnn.map_to_embedding(self.input_holders)
                # Call TF api to create recurrent neural network
                outputs, self.final_state = \
                    tf.nn.dynamic_rnn(self.cell, self.inputs, initial_state=self.state, time_major=True)
                # Reshape outputs and targets into [batch_size * num_steps, feature_dims]
                self.outputs = tf.reshape(outputs, [-1, outputs.get_shape().as_list()[2]])
                self.targets = tf.reshape(self.target_holders, [-1])
                if rnn.has_projcet:
                    # rnn has output project, do manual projection for speed
                    self.projected_outputs = rnn.project_output(self.outputs)
                    self.loss = rnn.loss_func(self.projected_outputs, self.targets)
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

    def run(self, inputs, targets, epoch_size, sess, run_ops=None, eval_ops=None, sum_ops=None, verbose=False):
        """
        RNN Model's main API for running the model from inputs
        Note: this function can only be run after the graph is finalized.
        When inputs and targets are numpy.ndarray, the epoch_size should be 1,
            or the model will run on the same data again and again.
        :param inputs: a input Feeder instance, a auto produce Tensor, or a numpy.ndarray
        :param targets: should be same type as inputs.
        :param epoch_size: how many iterations that the looping will be run (the size of the input queue)
        :param sess: the tf.Session to run the model
        :param run_ops: a dict containing all the ops that will be run but not logged
        :param eval_ops: a dict containing all the ops that will be logged but not verbosed
        :param sum_ops: a dict containing all the ops that will be verbosed. All ops in this dict should be scalar.
        :param verbose:
        :return: a pair of dicts (evals, sums), each containing running results of eval_ops and sum_ops,
            result of each op in the dict is stored as a list.
        """

        if run_ops is None: run_ops = {}
        if eval_ops is None: eval_ops = {}
        if sum_ops is None: sum_ops = {}
        verbose_every = epoch_size//10 if verbose else math.inf
        # initialize state and add ops that must be run
        if self.current_state is None:
            self.init_state(sess)
        # for compatible with different input type
        if isinstance(inputs, tf.Tensor):
            get_data = lambda x: sess.run(x)
        elif isinstance(inputs, Feeder):
            get_data = lambda x: [y() for y in x]
        elif isinstance(inputs, np.ndarray):
            get_data = lambda x: x
        else:
            raise TypeError("inputs mal format!")

        run_ops['state'] = self.final_state
        run_ops.update(eval_ops)
        run_ops.update(sum_ops)
        # total_loss = 0
        # vals = None
        start_time = verbose_time = time.time()
        evals = {name: [] for name in eval_ops}
        sums = {name: 0 for name in sum_ops}
        for i in range(epoch_size):
            feed_dict = self.feed_state(self.current_state)
            _inputs, _targets = get_data((inputs, targets))
            feed_dict[self.input_holders] = _inputs
            if _targets is not None:  # when we just need infer, we don't need to have targets
                feed_dict[self.target_holders] = _targets

            vals = sess.run(run_ops, feed_dict)
            self.current_state = vals['state']

            if eval_ops:
                for name in eval_ops:
                    evals[name].append(vals[name])
            if sum_ops:
                for name in sum_ops:
                    sums[name] += vals[name]
            if i % verbose_every == 0 and i != 0:
                delta_time = time.time() - verbose_time
                print("epoch[{:d}/{:d}] speed:{:.1f} wps, time: {:.1f}s".format(
                    i, epoch_size, i * self.batch_size * self.num_steps / (time.time() - start_time), delta_time))
                sum_list = ["{:s}: {:.4f}".format(name, value / i) for name, value in sums.items()]
                if sum_list: print(', '.join(sum_list))
                verbose_time = time.time()
        total_time = time.time() - start_time
        # Prepare for returning values
        # vals['loss'] = total_loss / epoch_size
        if verbose:
            sum_str = ', '.join(["{:s}: {:.4f}".format(name, value / i) for name, value in sums.items()])
            if sum_str: sum_str = ', '+sum_str
            print("Epoch Summary: total time:{:.1f}s, speed:{:.1f} wps".format(
                total_time, epoch_size * self.num_steps * self.batch_size / total_time) + sum_str)
        return evals, sums

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

    # def feed_data(self, data, inputs=True):
    #     holders = self.input_holders if inputs else self.target_holders
    #     return {holders: data}

    def init_state(self, sess):
        self.current_state = sess.run(self.state, )
        return self.current_state

    def log_ops_placement(self, ops, log_path):
        # Print ops assignments for debugging
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        _hdlr = tf.logging._handler
        new_handler = logging.FileHandler(log_path, 'a+')
        tf.logging._logger.removeHandler(_hdlr)
        tf.logging._logger.addHandler(new_handler)
        try:
            sess.run(ops)
        except:
            print("ops placements end.")
        tf.logging._logger.removeHandler(new_handler)
        tf.logging._logger.addHandler(_hdlr)
        print("ops placements info logged to {}".format(log_path))

    def do_projection(self, outputs, sess):
        """
        Project the cell outputs on to word space as a probability distribution
        :param outputs: a numpy array of shape [output_steps, feature_dims] as feed in data
        :param sess: the TF Session to run the computation
        :return: a numpy array of shape [output_steps, vocab_size] as the projected probability distribution
        """
        if not hasattr(self, 'projected_outputs'):
            projected_outputs = outputs
        else:
            projected = []
            batch_size = self.num_steps * self.batch_size  # the output Tensor has shape [num_steps*batch_size, dims]
            output_steps = outputs.shape[0]
            for begin in range(0, output_steps, batch_size):
                end = begin + batch_size
                if end > output_steps:
                    _projected = sess.run(self.projected_outputs, {self.outputs: outputs[-batch_size:,:]})
                    _projected = _projected[end-output_steps:,:]  # throw duplicated part
                else:
                    _projected = sess.run(self.projected_outputs, {self.outputs: outputs[begin:end, :]})
                projected.append(_projected)
            projected_outputs = np.vstack(projected)

        # do softmax
        return softmax(projected_outputs, axis=1)


class RNN(object):
    """
    A helper class that wraps TF RNNCells
    This class is used for wrapping RNN Model definition. You cannot use this class directly for computation.
    For computation (training, evaluating), use RNN.unroll() to create RNNModel,
    which create TF computation Graph for computation.
    """
    def __init__(self, name="RNN", initializer=None, logdir=None, graph=None):
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
        self.evaluator = None
        self.validator = None
        self.generator = None
        self.loss_func = None
        self.is_compiled = False
        self.embedding_size = None
        self.vocab_size = None
        self.supervisor = None
        self.models = []
        self.graph = graph if isinstance(graph, tf.Graph) else tf.get_default_graph()
        self.logdir = logdir or name

    def set_input(self, dshape, dtype, vocab_size, embedding_size=None):
        """
        set the input data shape of the model
        :param dshape: tuple or a list, [batch_size, ...]
            typically [None] for word-id input,
            for word embeddings, should be [None, feature_dim]
        :param dtype: tensorflow data type
        :return: None
        """
        assert not self.is_compiled
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
        assert not self.is_compiled
        self.output_shape = list(dshape)
        self.output_dtype = dtype

    def set_target(self, dshape, dtype=tf.int32):
        assert not self.is_compiled
        self.target_shape = dshape
        self.target_dtype = dtype

    def set_loss_func(self, loss_func):
        """
        Set the loss function of the model
        :param loss_func: a function of form loss = loss_func(outputs, targets), where loss is a scalar Tensor
        :return: None
        """
        assert not self.is_compiled
        self.loss_func = loss_func

    def add_cell(self, cell, *args, **kwargs):
        """
        A helper function to use for create RNN cells under specific TensorFlow VariableScope
        :param cell: should be subclasses of tf.nn.rnn_cell.BasicRNNCell
        :param name: the name or scope used for managing the cell
        :param kwargs: the arguments used to create the cell
        :return: a cell managed under scope
        """
        assert not self.is_compiled
        self.cell_list.append(cell(*args, **kwargs))

    def compile(self, evaluate=True):
        """
        Compile the model. Should be called before training or running the model.
        Basically, this function just do checkings on model configurations,
            and create a Evaluator which contains an unrolled model
        :return: None
        """
        if self.is_compiled:  # In case of multiple compiles
            print("Already compiled!")
            return
        if self.input_shape is None or self.input_dtype is None:
            raise ValueError("input_shape or input_dtype is None, call set_input first!")
        if self.output_shape is None or self.output_dtype is None:
            raise ValueError("output_shape or output_dtype is None, call set_output first!")
        if self.target_shape is None or self.target_dtype is None:
            raise ValueError("target_shape or target_dtype is None, call set_target first!")
        if self.loss_func is None:
            raise ValueError("loss_func is None, call set_loss_func first!")
        # This operation creates no tf.Variables, no need for using variable scope
        self._cell = tf.nn.rnn_cell.MultiRNNCell(cells=self.cell_list)
        # All done
        self.is_compiled = True
        # Create a default evaluator
        if evaluate:
            with self.graph.as_default():
                with tf.device("/cpu:0"):
                    self.evaluator = Evaluator(self, batch_size=1)

    def unroll(self, batch_size, num_steps, keep_prob=None, name=None):
        """
        Unroll the RNN Cell into a RNN Model with fixed num_steps and batch_size
        :param batch_size: batch_size of the model
        :param num_steps: unrolled num_steps of the model
        :param keep_prob: keep probability in training (like dropout in CNN)
        :param name: name of the model
        :return:
        """
        assert self.is_compiled
        with self.graph.as_default():  # Ensure that the model is created under the managed graph
            return RNNModel(self, batch_size, num_steps, keep_prob=keep_prob, name=name)

    def add_trainer(self, batch_size, num_steps, keep_prob=1.0, optimizer="GradientDescent", learning_rate=1.0, clipper=None):
        """
        The previous version of train is too messy, explicitly break them into add_trainer, and train
        add_trainer does all the preparations on building TF graphs, this function should be called before train
        :param batch_size: batch_size of the running
        :param num_steps: unrolled num_steps
        :param keep_prob: see unroll
        :param optimizer: the optimizer to use, can be a str indicating the tf optimizer, or a subclass of tf.train.Optimizer
        :param learning_rate:
        :param clipper:
        :return:
        """
        assert self.is_compiled
        if self.trainer is not None:
            print("trainer is already exists! Currently do not support multi training!")
            return
        with self.graph.as_default():
            self.trainer = Trainer(self, batch_size, num_steps, keep_prob, optimizer, learning_rate, clipper)

    def add_validator(self, batch_size, num_steps):
        assert self.is_compiled
        if self.validator is not None:
            print("validator is already exists! Currently do not support multi training!")
            return
        with self.graph.as_default():
            self.validator = Evaluator(self, batch_size, num_steps, False, False, False)

    def add_evaluator(self, batch_size=1, record_every=1, log_state=True, log_input=True, log_output=True):
        """
        Explicitly add evaluator instead of using the default one. You must explicitly call compile(evaluate=False)
            before calling this function
        :param batch_size: default to be 1
        :param record_every: record frequency
        :param log_state: flag of state logging
        :param log_input: flag of input logging
        :param log_output: flag of output logging
        :param logdir: logging directory
        :return:
        """
        assert self.evaluator is None
        with self.graph.as_default():
            with tf.device("/cpu:0"):
                self.evaluator = Evaluator(self, batch_size, record_every, log_state, log_input, log_output)

    def add_generator(self, word_to_id):
        assert self.generator is None
        with self.graph.as_default():
            with tf.device("/cpu:0"):
                self.generator = Generator(self, word_to_id)

    def train(self, inputs, targets, epoch_size, epoch_num, valid_inputs=None, valid_targets=None,
              valid_epoch_size=None, verbose=True):
        """
        Training using given input and target data
        TODO: Clean up this messy function
        :param inputs: should be a list of input sequence, each element of the list is a input sequence specified by x.
        :param targets: should be of size [num_seq, seq_length, ...]
        :param epoch_size: the size of an epoch
        :param epoch_num: number of training epochs
        :param valid_inputs: Validation input data
        :param valid_targets: Validation target data
        :param valid_epoch_size: batch_size of validation set
        :param verbose: Print training information if True
        :return: None
        """
        assert self.is_compiled
        assert self.trainer is not None
        with self.graph.as_default():
            self.finalize()
            print("Start Running Train Graph")
            with self.sess as sess:
                if valid_inputs is None:
                    # Only needs to run training graph
                    self.trainer.train(sess, inputs, targets, epoch_size, epoch_num)
                else:
                    for i in range(epoch_num):
                        if verbose:
                            print("Epoch {}:".format(i))
                        self.trainer.train_one_epoch(sess, inputs, targets, epoch_size, verbose=verbose)
                        self.validator.evaluate(sess, valid_inputs, valid_targets, valid_epoch_size, verbose=False)

    def evaluate(self, inputs, targets, epoch_size, logdir=None):
        assert self.is_compiled
        with self.graph.as_default():
            self.finalize()
            with self.sess as sess:
                logdir = os.path.join(self.logdir, "./evaluate") if logdir is None else logdir
                self.evaluator.evaluate(sess, inputs, targets, epoch_size, record=True, verbose=True, logdir=logdir)

    def generate(self, seed_id, logdir, max_branch=1, accum_cond_prob=0.9,
                 min_cond_prob=0.0, min_prob=0.0, max_step=20):
        assert self.is_compiled
        with self.graph.as_default():
            self.finalize()
            with self.sess as sess:
                return self.generator.generate(sess, seed_id, logdir, max_branch,
                                               accum_cond_prob, min_cond_prob, min_prob, max_step)

    def run_with_context(self, func, *args, **kwargs):
        assert self.is_compiled
        with self.graph.as_default():
            self.finalize()
            with self.sess as sess:
                func(sess, *args, **kwargs)

    def save(self, path=None):
        """
        Save the model to a given path
        :param path:
        :return:
        """
        if not self.finalized:
            self.finalize()
        path = path if path is not None else os.path.join(self.logdir, './model')
        with self.supervisor.managed_session() as sess:
            self.supervisor.saver.save(sess, path, global_step=self.supervisor.global_step)
            print("Model variables saved to {}.".format(path))

    def restore(self, path=None):
        if not self.finalized:
            self.finalize()
        path = path if path is not None else self.logdir
        checkpoint = tf.train.latest_checkpoint(path)
        print(path)
        print(checkpoint)
        with self.supervisor.managed_session() as sess:
            self.supervisor.saver.restore(sess, checkpoint)
        print("Model variables restored from {}.".format(path))

    def finalize(self):
        """
        After all the computation ops are built in the graph, build a supervisor which implicitly finalize the graph
        :return: None
        """
        if self.finalized:
            print("Graph has already been finalized!")
            return False
        self.supervisor = tf.train.Supervisor(self.graph, logdir=self.logdir)
        return True

    @property
    def finalized(self):
        return False if self.supervisor is None else True

    @property
    def cell(self):
        return self._cell

    @property
    def need_reuse(self):
        return None if len(self.models) == 0 else True

    @property
    def has_embedding(self):
        return bool(self.embedding_size)

    @property
    def has_projcet(self):
        return self.cell_list[-1].output_size != self.output_shape[-1]

    @property
    def sess(self):
        assert self.finalized
        return self.supervisor.managed_session(config=config_proto())

    def map_to_embedding(self, inputs):
        """
        Map the input ids into embedding
        :param inputs: a 2D Tensor of shape (num_steps, batch_size) of type int32, denoting word ids
        :return: a 3D Tensor of shape (num_Steps, batch_size, embedding_size) of type float32.
        """
        if self.has_embedding:
            # The Variables are already created in the compile(), need to
            with tf.variable_scope('embedding', initializer=self.initializer):
                with tf.device("/cpu:0"):  # Force CPU since GPU implementation is missing
                    embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=data_type())
                    return tf.nn.embedding_lookup(embedding, inputs)
        else:
            return None

    def project_output(self, outputs):
        """
        Project outputs into softmax distributions
        :param outputs: a tensor of shape (batch_size, output_size)
        :return: softmax distributions on vocab_size, a Tensor of shape (batch_size, vocab_size)
        """
        if self.has_projcet:
            with tf.variable_scope('project', initializer=self.initializer):
                project_w = tf.get_variable(
                    "project_w", [self.cell_list[-1].output_size, self.vocab_size], dtype=data_type())
                projcet_b = tf.get_variable("project_b", [self.vocab_size], dtype=data_type())
                return tf.matmul(outputs, project_w) + projcet_b
        else:
            return None


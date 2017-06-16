import tensorflow as tf

from rnnvis.vendor.seq2seq_model import Seq2SeqModel

from rnnvis.rnn.rnn import RNNModel, RNN, DropOutWrapper, MultiRNNCell, _input_and_global
from rnnvis.rnn.varlen_support import sequence_length, last_relevant
from rnnvis.rnn.command_utils import data_type
from rnnvis.rnn.eval_recorder import Recorder


class Seq2SeqEvaluator():

    def __init__(self, model, batch_size=1, num_steps=1, record_every=1, log_state=True, log_input=False,
                 log_output=True, log_gradients=False, log_pos=False, dynamic=True):
        """
        Create an unrolled rnn model with TF tensors
        :param rnn:
        :param batch_size:
        :param num_steps:
        :param keep_prob:
        :param name:
        """
        assert isinstance(model, Seq2SeqModel)
        self.model = model
        self._cell = rnn.cell
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.name = name or "UnRolled"
        self.dynamic = dynamic
        self.current_state = None
        # Ugly hacks for DropoutWrapper
        if keep_prob is not None and keep_prob < 1.0:
            cell_list = [DropOutWrapper(cell, output_keep_prob=keep_prob)
                         for cell in rnn.cell_list]
            self._cell = MultiRNNCell(cell_list, state_is_tuple=True)
        # The abstract name scope is not that easily dealt with, try to make the transparent to user
        with tf.name_scope(name):
            reuse = rnn.need_reuse
            with tf.variable_scope(rnn.name, reuse=reuse, initializer=rnn.initializer):
                # Build TF computation Graph
                input_shape = [batch_size] + [num_steps] + list(rnn.input_shape)[1:]
                # self.input_holders = tf.placeholder(rnn.input_dtype, input_shape, "input_holders")
                zero_initializer = tf.constant_initializer(value=0, dtype=rnn.input_dtype)
                self.input_holders = tf.Variable(zero_initializer(shape=input_shape), trainable=False,
                                                 collections=_input_and_global, name='input_holders')
                # self.input_holders = tf.Variable(np.zeros(input_shape))
                # self.batch_size = tf.shape(self.input_holders)[0]
                self.state = self.cell.zero_state(self.batch_size, rnn.output_dtype)
                # ugly hacking for EmbeddingWrapper Badness
                self.inputs = self.input_holders if not rnn.has_embedding \
                    else rnn.map_to_embedding(self.input_holders + 1)
                if keep_prob is not None and keep_prob < 1.0:
                    self.inputs = tf.nn.dropout(self.inputs, keep_prob)
                # Call TF api to create recurrent neural network
                self.input_length = sequence_length(self.inputs)
                if dynamic:
                    self.outputs, self.final_state = \
                        tf.nn.dynamic_rnn(self.cell, self.inputs, sequence_length=self.input_length,
                                          initial_state=self.state, dtype=data_type(), time_major=False)
                else:
                    inputs = [self.inputs[:, i] for i in range(num_steps)]
                    # Since we do not want it to be dynamic, sequence length is not fed,
                    # so that evaluator can fetch gate tensor values.
                    outputs, self.final_state = \
                        tf.nn.rnn(self.cell, inputs, initial_state=self.state, dtype=data_type())
                    self.outputs = tf.stack(outputs, axis=1)
                if rnn.use_last_output:
                    self.outputs = last_relevant(self.outputs, self.input_length)
                    target_shape = [batch_size] + list(rnn.target_shape)[1:]
                else:
                    target_shape = [batch_size] + [num_steps] + list(rnn.target_shape)[1:]
                # self.target_holders = tf.placeholder(rnn.target_dtype, target_shape, "target_holders")
                zero_initializer = tf.constant_initializer(value=0, dtype=rnn.target_dtype)
                self.target_holders = tf.Variable(zero_initializer(shape=target_shape), trainable=False,
                                                  collections=_input_and_global, name='target_holders')
                if rnn.has_project:
                    # Reshape outputs and targets into [batch_size * num_steps, feature_dims]
                    outputs = tf.reshape(self.outputs, [-1, self.outputs.get_shape().as_list()[-1]])
                    targets = tf.reshape(self.target_holders, [-1])
                    # rnn has output project, do manual projection for speed
                    self.projected_outputs = rnn.project_output(outputs)
                    self.loss = rnn.loss_func(self.projected_outputs, targets)
                    self.accuracy = tf.reduce_mean(tf.cast(
                        tf.nn.in_top_k(self.projected_outputs, targets, 1), data_type()))
                else:
                    self.projected_outputs = tf.reshape(self.outputs, [-1, self.outputs.get_shape().as_list()[-1]])
                    self.loss = rnn.loss_func(self.outputs, self.target_holders)
                    self.accuracy = tf.reduce_mean(tf.cast(
                        tf.nn.in_top_k(self.outputs, self.target_holders, 1), data_type()))

        # Append self to rnn's model list
        rnn.models.append(self)
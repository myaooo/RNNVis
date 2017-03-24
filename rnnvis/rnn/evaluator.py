"""
Evaluator Class
"""

from collections import defaultdict
from functools import reduce

import tensorflow as tf
import numpy as np

from rnnvis.rnn import rnn
from rnnvis.datasets.data_utils import Feeder
from rnnvis.rnn.eval_recorder import Recorder


tf.GraphKeys.EVAL_SUMMARIES = "eval_summarys"
_evals = [tf.GraphKeys.EVAL_SUMMARIES]


class Evaluator(object):
    """
    An evaluator evaluates a trained RNN.
    This class also provides several utilities for recording hidden states
    """

    def __init__(self, rnn_, batch_size=1, num_steps=1, record_every=1, log_state=True, log_input=False, log_output=True,
                 log_gradients=False, log_gates=False, log_pos=False, dynamic=True):
        assert isinstance(rnn_, rnn.RNN)
        self._rnn = rnn_
        self._record_every = record_every
        self.log_state = log_state
        self.log_input = log_input
        self.log_output = log_output
        self.log_gradients = log_gradients
        self.log_gates = log_gates
        self.log_pos = log_pos
        self.model = rnn_.unroll(batch_size, num_steps, name='EvaluateModel{:d}'.format(len(rnn_.models)),
                                 dynamic=dynamic)
        summary_ops = defaultdict(list)
        if log_state:
            for s in self.model.final_state:
                # s is tuple
                if isinstance(s, tf.nn.rnn_cell.LSTMStateTuple):
                    summary_ops['state_c'].append(s.c)
                    summary_ops['state_h'].append(s.h)
                else:
                    summary_ops['state'].append(s)
            for name, states in summary_ops.items():
                # states is a list of tensor of shape [batch_size, n_units],
                # we want the stacked shape to be [batch_size, n_layer, n_units]
                summary_ops[name] = tf.stack(states, axis=1)
        if log_input:
            summary_ops['input'] = self.model.input_holders
            if rnn_.map_to_embedding:
                summary_ops['input_embedding'] = self.model.inputs
        if log_output:
            summary_ops['output'] = self.model.outputs
        if log_gradients:
            inputs_gradients = tf.gradients(self.model.loss, self.model.inputs)
            summary_ops['inputs_gradients'] = inputs_gradients
        if log_gates:
            gates = self.model.get_gate_tensor()
            if gates is None:
                print("WARN: No gates tensor available, Are you using RNN?")
            else:
                # inputs = self.model.inputs
                gate_ops = defaultdict(list)
                for gate in gates:
                    if isinstance(gate, tuple):  # LSTM gates are a tuple of (i, f, o)
                        gate_ops['gate_i'].append(tf.sigmoid(gate[0]))
                        gate_ops['gate_f'].append(tf.sigmoid(gate[1]))
                        gate_ops['gate_o'].append(tf.sigmoid(gate[2]))
                    else: # GRU only got one gate z
                        gate_ops['gate'].append(gate)
                for name, gate in gate_ops.items():
                    # states is a list of tensor of shape [batch_size, n_units],
                    # we want the stacked shape to be [batch_size, n_layer, n_units]
                    summary_ops[name] = tf.stack(gate, axis=1)
                # summary_ops.update(gate_ops)
        self.summary_ops = summary_ops
        self.pos_tagger = None
        if log_pos:
            if self._rnn.id_to_word is None:
                raise ValueError('Evaluator: RNN instance needs to have id_to_word property in order to log_pos!')
            import nltk  # lazy import

            def tagger(ids):
                tokens = self._rnn.get_word_from_id(ids)
                if len(tokens) != len(ids):
                    raise ValueError('Evaluator: tokens length {:d} and ids length {:d} mismatch'
                                     .format(len(tokens), len(ids)))
                tokens_tags = nltk.pos_tag(tokens, tagset='universal', lang='eng')
                _, tags = zip(*tokens_tags)
                return tags
            self.pos_tagger = tagger

    @property
    def record_every(self):
        return self._record_every

    @record_every.setter
    def record_every(self, v):
        assert isinstance(v, int), "record_every should be an integer!"
        self._record_every = v

    def evaluate(self, sess, inputs, targets, input_size, verbose=True, refresh_state=False):
        """
        Evaluate on the test or valid data
        :param inputs: a Feeder instance
        :param targets: a Fedder instance
        :param input_size: size of the input
        :param sess: tf.Session to run the computation
        :param verbose: verbosity
        :param refresh_state: True if you want to refresh hidden state after each loop
        :return:
        """

        self.model.reset_state()
        # eval_ops = self.summary_ops
        sum_ops = {"loss": self.model.loss, 'acc-1': self.model.accuracy}
        # loss = 0
        # acc = 0
        # print("Start evaluating...")
        # for i in range(0, input_size):
        evals, sums = self.model.run(inputs, targets, input_size, sess, sum_ops=sum_ops,
                                     verbose=False, refresh_state=refresh_state)
        loss = sums["loss"] / input_size
        acc = sums['acc-1'] / input_size

        if verbose:
            print("Evaluate Summary: acc-1: {:.4f}, avg loss:{:.4f}".format(acc, loss), flush=True)
        return loss, acc

    def evaluate_and_record(self, sess, inputs, targets, recorder, verbose=True, refresh_state=False):
        """
        A similar method like evaluate.
        Evaluate model's performance on a sequence of inputs and targets,
        and record the detailed information with recorder.
        :param inputs: an object convertible to a numpy ndarray, with 2D shape [batch_size, length],
            elements are word_ids of int type
        :param targets: same as inputs, no loss will be calculated if targets is None
        :param sess: the sess to run the computation
        :param recorder: an object with method `start(inputs, targets)` and `record(record_message)`
        :param verbose: verbosity
        :return:
        """

        assert isinstance(inputs, Feeder), 'expect inputs type Feeder but got type {:s}'.format(str(type(inputs)))
        assert isinstance(targets, Feeder) or targets is None
        assert isinstance(recorder, Recorder), "recorder should be an instance of rnn.eval_recorder.Recorder!"
        recorder.start(inputs, targets, self.pos_tagger)
        input_size = inputs.epoch_size
        print("input size: {:d}".format(input_size))
        eval_ops = self.summary_ops
        self.model.reset_state()
        for i in range(0, input_size, self.record_every):
            if refresh_state:
                self.model.reset_state()
            n_steps = input_size - i if i + self.record_every > input_size else self.record_every

            evals, _ = self.model.run(inputs, targets, n_steps, sess, eval_ops=eval_ops,
                                      verbose=False, refresh_state=False)
            messages = [{name: value[i] for name, value in evals.items()} for i in range(n_steps)]
            for message in messages:
                recorder.record(message)
            if verbose and (i//self.record_every + 1) % (input_size // self.record_every // 10) == 0:
                print("[{:d}/{:d}] completed".format(i+self.record_every, input_size), flush=True)
        recorder.flush()
        print("Evaluation done!")

    def _cal_salience(self, sess, embedding=None, feed_dict=None, y_or_x=None):
        """
        Calculate the saliency matrix of states regarding inputs,
        this should be called on a trained model for evaluation
        (you can also call this on a just initialized one to compare)
        :param sess: the sess to run the computation
        :param embedding: the word embedding to put into the computation of the gradients
        :param feed_dict: extra feed_dict, should feed in the states
        :return:
        """
        salience = defaultdict(list)
        inputs = self.model.inputs
        if feed_dict is None:
            self.model.init_state(sess)
            feed_dict = self.model.feed_state(self.model.current_state)
        if isinstance(embedding, int):
            embedding = sess.run(inputs, {self.model.input_holders: np.array([embedding]).reshape(1, 1)})
        with sess.as_default():
            for state in self.model.final_state:
                if isinstance(state, tf.nn.rnn_cell.LSTMStateTuple):
                    salience['state_c'].append(cal_jacobian(state.c, inputs, embedding, feed_dict, y_or_x))
                    salience['state_h'].append(cal_jacobian(state.h, inputs, embedding, feed_dict, y_or_x))
                else:
                    salience['state'].append(cal_jacobian(state, inputs, embedding, feed_dict, y_or_x))
            gates = self.model.get_gate_tensor()
            if gates is None:
                return salience
            for gate in gates:
                if isinstance(gate, tuple):  # LSTM gates are a tuple of (i, f, o)
                    salience['gate_i'].append(cal_jacobian(gate[0], inputs, embedding, feed_dict, y_or_x))
                    salience['gate_f'].append(cal_jacobian(gate[1], inputs, embedding, feed_dict, y_or_x))
                    salience['gate_o'].append(cal_jacobian(gate[2], inputs, embedding, feed_dict, y_or_x))
                else:
                    salience['gate'].append(cal_jacobian(gate, inputs, embedding, feed_dict, y_or_x))

        return salience

    def cal_salience(self, sess, word_ids, feed_dict=None, y_or_x=None, verbose=True):
        """
        This function calculate the salience of states and gates w.r.t. given words.
        The method used will not add any new TF ops, feel free to finalize the graph before calling this function
        Note: this function is very time consuming. About 5s per word on a laptop
        :param sess: the session to run the computation
        :param word_ids: the word_ids as a list
        :param feed_dict: additional feed_dict to feed in the sess.run()
        :param y_or_x: see docs of cal_gradients
        :param verbose: print progress
        :return: a list of salience
        """
        if isinstance(word_ids, int):
            word_ids = [word_ids]
        elif not isinstance(word_ids, list):
            raise TypeError("word_ids should be of type int of a list of int, but it's of type: {:s}"
                            .format(str(type(word_ids))))
        saliences = []
        for i, word in enumerate(word_ids):
            saliences.append(self._cal_salience(sess, word, feed_dict, y_or_x))
            if (i+1) % 20 == 0 and verbose:
                print("{:d}/{:d} completed".format(i+1, len(word_ids)))
        print("salience computation finished.")
        return saliences


def cal_jacobian(y, x, x_val=None, feed_dict=None, y_or_x=None):
    """
    A numerical way of calculating the Jacobian Matrix of y w.r.t to x
    :param y:
    :param x:
    :param x_val:
    :param feed_dict:
    :param y_or_x: if None, do not do any projection, directly return the Jacobian matrix,
        if 'y', return a 1-D vector of length y_len, which is the sum(dy/dx) over x
        if 'x', return a 1-D vector of length x_len, which is the sum(dy/dx) over y
    :return:
    """
    delta = 1e-7
    x_shape = x.get_shape().as_list()
    x_len = reduce(lambda a, b: a*b, x_shape)
    if x_val is None:
        x_val = np.zeros((x_len, ), x.dtype.as_numpy.dtype)
    else:
        x_val = x_val.reshape((x_len, ))
    if feed_dict is None:
        feed_dict = {}

    _jacobian = []
    for i in range(x_len):
        input_x = x_val.copy()

        input_x[i] += delta
        feed_dict[x] = input_x.reshape(x_shape)
        y_val1 = y.eval(feed_dict, tf.get_default_session()).reshape(-1)
        input_x[i] -= 2*delta
        feed_dict[x] = input_x.reshape(x_shape)
        y_val2 = y.eval(feed_dict, tf.get_default_session()).reshape(-1)
        _jacobian.append((y_val1 - y_val2))

    jacobian = (np.stack(_jacobian) / (2*delta)).T  # a correct jacobian should has shape [y_len, x_len]
    if y_or_x is None:
        return jacobian
    elif y_or_x is 'x':  # ones(y_len) * jacobian => shape: [x_len,]
        return np.sum(jacobian, axis=0)
    elif y_or_x is 'y':  # jacobian * ones(x_len) => shape: [y_len,]
        return np.sum(jacobian, axis=1)

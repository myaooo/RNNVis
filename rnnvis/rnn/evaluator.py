"""
Evaluator Class
"""

from collections import defaultdict
from functools import reduce

import tensorflow as tf
import numpy as np

from rnnvis.rnn import rnn
from rnnvis.db.language_model import insert_evaluation, push_evaluation_records
from rnnvis.datasets.data_utils import InputProducer, Feeder


tf.GraphKeys.EVAL_SUMMARIES = "eval_summarys"
_evals = [tf.GraphKeys.EVAL_SUMMARIES]


class Evaluator(object):
    """
    An evaluator evaluates a trained RNN.
    This class also provides several utilities for recording hidden states
    """

    def __init__(self, rnn_, batch_size=1, num_steps=1, record_every=1, log_state=True, log_input=True, log_output=True,
                 log_gradients=False, log_gates=False, dynamic=True):
        assert isinstance(rnn_, rnn.RNN)
        self._rnn = rnn_
        self.record_every = record_every
        self.log_state = log_state
        self.log_input = log_input
        self.log_output = log_output
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
                inputs = self.model.inputs
                for gate in gates:
                    if isinstance(gate, tuple):  # LSTM gates are a tuple of (i, f, o)
                        summary_ops['gate_i'].append(tf.gradients(gate[0], inputs))
                        summary_ops['gate_f'].append(tf.gradients(gate[1], inputs))
                        summary_ops['gate_o'].append(tf.gradients(gate[2], inputs))
                    else: # GRU only got one gate z
                        summary_ops['gate'].append(tf.gradients(gate, inputs))
        self.summary_ops = summary_ops

        # adding salience computations
        # self.salience = self._init_salience_ops() if cal_salience else None

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
        eval_ops = self.summary_ops
        sum_ops = {"loss": self.model.loss, 'acc-1': self.model.accuracy}
        loss = 0
        acc = 0
        # print("Start evaluating...")
        for i in range(0, input_size, self.record_every):
            evals, sums = self.model.run(inputs, targets, self.record_every, sess, eval_ops=eval_ops, sum_ops=sum_ops,
                                         verbose=False, refresh_state=refresh_state)
            loss += sums["loss"]
            acc += sums['acc-1']
            # if i % 500 == 0 and i != 0:
            #     if verbose:
            #         print("[{:d}/{:d}]: avg loss:{:.3f}".format(i, input_size, loss/(i+1)))
        loss /= (input_size * self.record_every)
        acc /= (input_size * self.record_every)
        if verbose:
            print("Evaluate Summary: acc-1: {:.4f}, avg loss:{:.4f}".format(acc, loss), flush=True)

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

        assert isinstance(inputs, Feeder)
        assert isinstance(targets, Feeder) or targets is None
        recorder.start(inputs, targets)
        input_size = inputs.epoch_size
        print("input size: {:d}".format(input_size))
        eval_ops = self.summary_ops
        self.model.reset_state()
        for i in range(0, input_size):
            evals, _ = self.model.run(inputs, targets, self.record_every, sess, eval_ops=eval_ops,
                                      verbose=False, refresh_state=refresh_state)
            messages = [{name: value[i] for name, value in evals.items()} for i in range(self.record_every)]
            for message in messages:
                recorder.record(message)
            if verbose and i % (input_size // 10) == 0 and i != 0:
                print("[{:d}/{:d}] completed".format(i, input_size), flush=True)
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


class Recorder(object):

    def start(self, inputs, targets):
        """
        prepare the recording
        :param inputs: should be an instance of data_utils.Feeder
        :param targets: should be an instance of data_utils.Feeder or None
        :return: None
        """
        raise NotImplementedError("This is the Recorder base class")

    def record(self, message):
        """
        Do some preparations on the message, and then do the recording
        :param message: a dictionary, containing the results of a run of the loo[
        :return: None
        """
        raise NotImplementedError("This is the Recorder base class")

    def flush(self):
        """
        Used for flushing the records to the disk / db
        :return: None
        """
        raise NotImplementedError("This is the Recorder base class")


class StateRecorder(Recorder):

    def __init__(self, data_name, model_name, flush_every=100):
        self.data_name = data_name
        self.model_name = model_name
        self.eval_doc_id = []
        self.buffer = defaultdict(list)
        self.batch_size = 1
        self.inputs = None
        self.flush_every = flush_every
        self.step = 0

    def start(self, inputs, targets):
        """
        prepare the recording
        :param inputs: should be an instance of data_utils.Feeder
        :param targets: should be an instance of data_utils.Feeder or None
        :return: None
        """
        self.batch_size = inputs.shape[0]
        self.inputs = inputs.full_data
        for i in range(inputs.shape[0]):
            self.eval_doc_id.append(insert_evaluation(self.data_name, self.model_name, self.inputs[i].tolist()))

    def record(self, record_message):
        """
        Record one step of information, note that there is a batch of them
        :param record_message: a dict, with keys as summary_names,
            and each value as corresponding record info [batch_size, ....]
        :return:
        """
        records = [{name: value[i] for name, value in record_message.items()} for i in range(self.batch_size)]
        for i, record in enumerate(records):
            record['word_id'] = int(self.inputs[i, self.step])
        self.buffer['records'] += records
        self.buffer['eval_ids'] += self.eval_doc_id
        self.step += 1
        if len(self.buffer['eval_ids']) >= self.flush_every:
            self.flush()

    def flush(self):
        push_evaluation_records(self.buffer.pop('eval_ids'), self.buffer.pop('records'))


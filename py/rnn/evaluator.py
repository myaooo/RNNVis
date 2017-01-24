"""
Evaluator Class
"""

from collections import  namedtuple
# from py.rnn.command_utils import *
import tensorflow as tf
from . import rnn


tf.GraphKeys.EVAL_SUMMARIES = "eval_summarys"
_evals = [tf.GraphKeys.EVAL_SUMMARIES]


class GenerateNode(object):
    """
    Node structure to store generation tree of a RNN
    """
    def __init__(self, word_id, prob, next_nodes=None):
        self.word_id = word_id
        self.prob = prob
        self.next_nodes = next_nodes

    def add_next_node(self, node):
        assert isinstance(node, GenerateNode)
        self.next_nodes.append(node)


class Evaluator(object):
    """
    An evaluator evaluates a trained RNN.
    This class also provides several utilities for recording hidden states
    """

    def __init__(self, rnn_, batch_size=1, record_every=1, log_state=True, log_input=True, log_output=True):
        assert isinstance(rnn_, rnn.RNN)
        self.record_every = record_every
        self.log_state = log_state
        self.log_input = log_input
        self.log_output = log_output
        self.model = rnn_.unroll(batch_size, record_every, name='EvaluateModel')
        self.summary_ops = []
        # self.logdir = logdir if logdir is not None else rnn_.logdir
        # self.writer = tf.summary.FileWriter(self.logdir)
        if log_state:
            for i, s in enumerate(self.model.state):
                # s is tuple
                if isinstance(s, tf.nn.rnn_cell.LSTMStateTuple):
                    self.summary_ops.append(tf.summary.tensor_summary("state_layer_{}_c".format(i), s.c, collections=_evals))
                    self.summary_ops.append(tf.summary.tensor_summary("state_layer_{}_h".format(i), s.h, collections=_evals))
                else:
                    self.summary_ops.append(tf.summary.tensor_summary("state_layer_{}".format(i), s, collections=_evals))
        if log_input:
            self.summary_ops.append(tf.summary.tensor_summary("input", self.model.input_holders, collections=_evals))
            if rnn_.map_to_embedding:
                self.summary_ops.append(tf.summary.tensor_summary("input_embedding", self.model.inputs, collections=_evals))
        if log_output:
            self.summary_ops.append(tf.summary.tensor_summary("output", self.model.outputs, collections=_evals))
        if len(self.summary_ops) == 0:
            self.merged_summary = None
        else:
            self.merged_summary = tf.summary.merge(self.summary_ops, _evals, "eval_summaries")

    def evaluate(self, inputs, targets, input_size, sess, record=False, verbose=True, logdir=None):
        if record:
            writer = tf.summary.FileWriter(logdir)
        else:
            writer = None
        self.model.init_state(sess)
        eval_ops = {"summary": self.merged_summary} if self.merged_summary is not None else {}
        sum_ops = {"loss": self.model.loss}
        total_loss = 0
        print("Start evaluating...")
        for i in range(input_size):
            evals, sums = self.model.run(inputs, targets, 1, sess, eval_ops=eval_ops, sum_ops=sum_ops, verbose=False)
            if record and eval_ops:
                summary = evals["summary"][0]
                writer.add_summary(summary, i*self.record_every)
            total_loss += sums["loss"]
            if i % 500 == 0:
                if verbose:
                    print("[{:d}/{:d}]: avg loss:{:.3f}".format(i, input_size, total_loss/(i+1)))
        if record:
            writer.close()
        loss = total_loss / input_size
        print("Evaluate Summary: avg loss:{:.3f}".format(loss))

    def generate(self, sess, seed, logdir, branch_num=3, accum_prob=0.9, neglect_prob=0.05, max_step=20):
        writer = tf.summary.FileWriter(logdir)
        model = self.model
        model.init_state(sess)
        def _generate(inputs, step):
            outputs = model.run(inputs, )


        tree = _generate(seed, max_step)
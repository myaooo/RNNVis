"""
Evaluator Class
"""

import tensorflow as tf
from . import rnn


class EvaluateConfig(object):
    """
    Configurations for Evaluation
    """
    def __init__(self, save_path, state, output, ):
        self.save_path = save_path
        self.state


class Evaluator(object):
    """
    An evaluator evaluates a trained RNN.
    This class also provides several utilities for recording hidden states
    """
    def __init__(self, rnn_, batch_size, log_period, state=True, input=True, output=True):
        self.model = rnn_.unroll(batch_size, config.state, name='Evaluate')
        self.config = config
        self._init_state = None
        self._final_state = None

    def evaluate(self, inputs, targets, verbose=True):
        pass
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


class Evaluator(object):
    """
    An evaluator evaluates a trained RNN.
    This class also provides several utilities for recording hidden states
    """
    def __init__(self, model, config):
        self.rnn = model
        self.config = config
        self._init_state = None
        self._final_state = None

    def evaluate(self, inputs, targets, verbose=True):
        pass
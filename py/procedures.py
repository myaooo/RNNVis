"""
Pre-defined procedures for running rnn, evaluating and recording
"""


import os
import tensorflow as tf
from rnn.config_utils import RNNConfig, TrainConfig
from rnn.command_utils import data_type
from rnn.rnn import RNN


def build_rnn(rnn_config):
    """
    Build an instance of RNN from config
    :param rnn_config: a RNNConfig instance
    :return: a compiled model
    """
    assert isinstance(rnn_config, RNNConfig)
    _rnn = RNN(rnn_config.name, rnn_config.initializer, os.path.join('models/', rnn_config.name))
    _rnn.set_input([None], rnn_config.input_dtype, rnn_config.vocab_size, rnn_config.embedding_size)
    for cell in rnn_config.cells:
        _rnn.add_cell(rnn_config.cell, **cell)
    _rnn.set_output([None, rnn_config.vocab_size], data_type())
    _rnn.set_target([None], rnn_config.target_dtype)
    _rnn.set_loss_func(rnn_config.loss_func)
    _rnn.compile()
    return _rnn


def build_trainer(rnn_, train_config):
    """
    Add a trainer to an already compiled RNN instance
    :param rnn_: the already compiled RNN
    :param train_config: a TrainConfig instance
    :return: the rnn_ with trainer added
    """
    assert isinstance(rnn_, RNN)
    assert isinstance(train_config, TrainConfig)
    rnn_.add_trainer(train_config.batch_size, train_config.num_steps, train_config.keep_prob, train_config.optimizer,
                     train_config.lr, train_config.clipper)
    rnn_.add_validator(train_config.batch_size, train_config.num_steps)


def build_model(config, train=True):
    """
    A helper function that wraps build_rnn and build_train to build a model
    :param config:
    :param train:
    :return:
    """
    if isinstance(config, str):
        rnn_config = RNNConfig.load(config)
        if train:
            train_config = TrainConfig.load(config)
    elif isinstance(config, dict):
        rnn_config = config['model']
        if train:
            train_config = config['train']
    else:
        raise TypeError('config should be a file_path or a dict!')
    rnn_ = build_rnn(rnn_config)
    if train:
        build_trainer(rnn_, train_config)
        return rnn_, train_config
    return rnn_

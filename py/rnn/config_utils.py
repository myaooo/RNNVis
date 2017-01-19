"""
Helpers for running rnn services from configurations
"""


import yaml
import tensorflow as tf
from . import rnn


__str2initializer = {
    'random_uniform': tf.random_uniform_initializer,
    'constant': tf.constant_initializer,
    'truncated_normal': tf.truncated_normal_initializer
}


def get_initializer(initializer, **kwargs):
    if callable(initializer):
        return initializer(**kwargs)
    if isinstance(initializer, str):
        if initializer in __str2initializer:
            return __str2initializer[initializer](**kwargs)
        else:
            print("{} is not an available initializer, default to tf.random_uniform_initializer".format(initializer))
    return tf.random_uniform_initializer(**kwargs)


__str2dtype = {
    'int32': tf.int32,
    'float32': tf.float32
}


def get_dtype(dtype):
    assert isinstance(dtype, str)
    if dtype in __str2dtype:
        return __str2dtype[dtype]
    return tf.int32


__str2cell = {
    'BasicLSTM': tf.nn.rnn_cell.BasicLSTMCell,
    'BasicRNN': tf.nn.rnn_cell.BasicRNNCell,
    'LSTM': tf.nn.rnn_cell.LSTMCell,
    'GRU': tf.nn.rnn_cell.GRUCell
}


def getRNNCell(cell):
    if isinstance(cell, tf.nn.rnn_cell.RNNCell):
        return cell
    assert isinstance(cell, str)
    if cell in __str2cell:
        return __str2cell[cell]
    return tf.nn.rnn_cell.BasicLSTMCell


def get_loss_func(loss_func):
    if callable(loss_func):
        return loss_func
    assert isinstance(loss_func, str)
    if loss_func == 'sequence_loss':
        return rnn.sequence_loss
    return rnn.sequence_loss


class RNNConfig(object):
    """
    Helper Class to create a RNN model
    """
    def __init__(self, name='RNN', initializer_name=None, initializer_args=None, vocab_size=1000, embedding_size=50,
                 input_dtype='int32', target_dtype='int32', cells=None, cell_type='BasicLSTM',
                 loss_func='sequence_loss'):
        self.name = name
        self.cells = cells
        self.cell = getRNNCell(cell_type)
        self.initializer = get_initializer(initializer_name, **initializer_args)
        self.input_dtype = get_dtype(input_dtype)
        self.target_dtype = get_dtype(target_dtype)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.loss_func = get_loss_func(loss_func)


    @staticmethod
    def load(file_path):
        with open(file_path) as f:
            config_dict = yaml.safe_load(f)['model']
            return RNNConfig(**config_dict)


def build_rnn(config):
    """
    Build a RNN from config
    :param config:
    :return:
    """
    if isinstance(config, str):
        config = RNNConfig.load(config)
    assert isinstance(config, RNNConfig)
    model = rnn.RNN(config.name, config.initializer)
    model.set_input([None], config.input_dtype, config.vocab_size, config.embedding_size)
    for cell in config.cells:
        model.add_cell(config.cell, **cell)
    model.set_output([None, config.vocab_size], tf.float32)
    model.set_target([None], config.target_dtype)
    model.set_loss_func(config.loss_func)
    model.compile()
    return model


class TrainConfig(object):
    """Manage configurations for Training"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def load(file_path):
        with open(file_path) as f:
            config_dict = yaml.safe_load(f)['train']
            return TrainConfig(**config_dict)

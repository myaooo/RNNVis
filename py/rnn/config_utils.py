"""
Helpers for running rnn services from configurations
"""


import os
import yaml
import tensorflow as tf
from . import rnn
from . import trainer


__str2initializer = {
    'random_uniform': tf.random_uniform_initializer,
    'constant': tf.constant_initializer,
    'truncated_normal': tf.truncated_normal_initializer
}


def get_initializer(initializer, **kwargs):
    """
    A helper function to get TensorFlow variable initializer
    :param initializer: a tf.*_initializer, or a str denoting the name of the function
    :param kwargs: the input kwargs for the initializer function
    :return: a TF initializer
    """
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
    """
    A helper function to get tf.dtype from str
    :param dtype: a str, e.g. "int32"
    :return: corresponding tf.dtype
    """
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


def get_rnn_cell(cell):
    """
    A helper function to get tf RNNCell
    :param cell: a subclass of RNNCell or a str denoting existing implementations
    :return: a Cell class
    """
    if isinstance(cell, str):
        if cell in __str2cell:
            return __str2cell[cell]
    try:
        if issubclass(cell, tf.nn.rnn_cell.RNNCell):
            return cell
    finally:
        return tf.nn.rnn_cell.BasicLSTMCell


def get_loss_func(loss_func):
    """
    A helper class to get model loss function from str
    TODO: add wrappers for all the TF loss function
    :param loss_func: a str denoting the loss function
    :return: a callable loss_func
    """
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
        self.cell = get_rnn_cell(cell_type)
        self.initializer = get_initializer(initializer_name, **initializer_args)
        self.input_dtype = get_dtype(input_dtype)
        self.target_dtype = get_dtype(target_dtype)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.loss_func = get_loss_func(loss_func)

    @staticmethod
    def load(file_or_dict):
        """
        Load an RNNConfig from config file
        :param file_or_dict: path of the config file
        :return: an instance of RNNConfig
        """
        if isinstance(file_or_dict, dict):
            config_dict = file_or_dict['model']
        else:
            with open(file_or_dict) as f:
                try:
                    config_dict = yaml.safe_load(f)['model']
                except:
                    raise ValueError("Malformat of config file!")
        return RNNConfig(**config_dict)


def parse_lr_from_config(lr):
    if isinstance(lr, str):
        try:
            func = eval(lr)
        except:
            raise ValueError('If learning_rate is a str, it should be a lambda expression of form f(epoch) -> rate')

    elif isinstance(lr, dict):
        try:
            func = trainer.get_lr_decay(lr['decay'], **lr['decay_args'])
        except:
            raise ValueError('If learning_rate is a dict, it should has keys "decay" and "decay_args"')
    else:
        try:
            func = float(lr)
        except:
            raise ValueError('Mal-format for a input learning_rate, it should be a number, '
                             'a str of lambda expression, or a dict specifying tf learning_rate_decay')
    return func


class TrainConfig(object):
    """Manage configurations for Training"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.clipper = trainer.get_gradient_clipper(self.gradient_clip, **self.gradient_clip_args)
        if hasattr(self, 'optimizer_args'):
            self.optimizer = trainer.get_optimizer(self.optimizer, **self.optimizer_args)
        self.optimizer = trainer.get_optimizer(self.optimizer)
        self.lr = parse_lr_from_config(self.learning_rate)

    @staticmethod
    def load(file_or_dict):
        """
        Load an TrainConfig from config file
        :param file_path: path of the config file
        :return: an instance of TrainConfig
        """
        if isinstance(file_or_dict, dict):
            config_dict = file_or_dict['train']
        else:
            with open(file_or_dict) as f:
                try:
                    config_dict = yaml.safe_load(f)['train']
                except:
                    raise ValueError("Malformat of config file!")
        return TrainConfig(**config_dict)

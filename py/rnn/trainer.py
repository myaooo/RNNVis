"""
Trainer handling TensorFlow Graph training computations
"""

import tensorflow as tf
import time
from . import rnn
from .config import *


_str2optimizers = {
    "Adam": tf.train.AdamOptimizer,
    "Adagrad": tf.train.AdagradOptimizer,
    "Momentum": tf.train.MomentumOptimizer,
    "GradientDescent": tf.train.GradientDescentOptimizer
}


def get_optimizer(optimizer):
    """
    Simple helper to get TensorFlow Optimizer
    :param optimizer: a str specifying the name of the optimizer to use,
        or a callable function that is already a TF Optimizer
    :return: a TensorFlow Optimizer
    """
    if isinstance(optimizer, tf.train.Optimizer):
        return optimizer
    if optimizer in _str2optimizers:
        return _str2optimizers[optimizer]
    else:
        raise ValueError('optimizer should be an instance of tf.train.Optimizer or a key in _str2optimizer!')


_str2clipper = {
    'value': tf.clip_by_value,
    'norm': tf.clip_by_norm,
    'average_norm': tf.clip_by_average_norm,
    'global_norm': tf.clip_by_global_norm
}


def get_gradient_clipper(clipper, *args, **kwargs):
    """
    Simple helper to get Gradient Clipper
    E.g: clipper = get_gradient_clipper('value', value_min, value_max, name='ValueClip')
    :param clipper: a string denoting TF Gradient Clipper (e.g. "global_norm", denote tf.clip_by_global_norm)
        or a function of type f(tensor) -> clipped_tensor
    :param args: used to create the clipper
    :param kwargs: used to create the clipper
    :return: a function (tensor) -> (clipped tensor)
    """
    if callable(clipper):
        return clipper
    # workaround of global_norm clipper, since it returns two variable with the second one as a scalar tensor
    if clipper == 'global_norm':
        return lambda t: tf.clip_by_global_norm(t, *args, **kwargs)[0]
    if clipper in _str2clipper:
        clipper = _str2clipper[clipper]
    else:
        raise ValueError('clipper should be a callable function or a given key in _str2clipper!')
    return lambda t: clipper(t, *args, **kwargs)


_str2decay = {
    'exponential': tf.train.exponential_decay,
    'inverse_time': tf.train.inverse_time_decay,
    'natural_exp': tf.train.natural_exp_decay,
    'polynomial': tf.train.polynomial_decay
}


def get_lr_decay(decay, *args, **kwargs):
    """
    Get a more convenient learning rate decay function.
    E.g. decay_func = get_lr_decay("exponential", 0.1, decay_steps=10000, decay_rate=0.95)
    :param decay: a str in the keys of _str2decay, or a function of the form:
        f(global_step) -> current_learning_rate, where global_step is a Python number
    :return:
    """
    if callable(decay):
        return decay
    if not isinstance(decay, str):
        raise TypeError("The input argument should be a callable function or a str!")
    if decay == 'piecewise_constant':
        return lambda global_step: tf.train.piecewise_constant(global_step, *args, **kwargs)
    if decay in _str2decay:
        decay_func = _str2decay[decay]
        if len(args) == 0:
            return lambda global_step: decay_func(kwargs.pop('learning_rate'), global_step, **kwargs)
        else:
            return lambda global_step: decay_func(args[0], global_step, *args[1:], **kwargs)
    else:
        raise ValueError("Cannot find corresponding decay function for the input {}".format(decay))


class Trainer(object):
    """
    Trainer Class
    Usage:
        TODO
    """
    def __init__(self, rnn_, batch_size, num_steps, keep_prob, optimizer,
                 learning_rate=0.1, gradient_clipper=None, decay=None, valid_model=None):
        """
        :param model: a instance of RNNModel class, the rnn model to be trained
        :param optimizer: the optimizer used to minimize model.loss, should be instance of tf.train.Optimizer
        :param learning_rate: a Tensor denoting the learning_rate to use
        :param supervisor: the supervisor used to manage training and saving model checkpoints
        :param valid_model: a instance of RNNModel class, the model used to run on validation set
        """
        if not isinstance(rnn_, rnn.RNN):
            raise TypeError("rnn should be instance of RNN")
        self.model = rnn_.unroll(batch_size, num_steps, keep_prob, name="Train")
        self._lr = learning_rate
        self.optimizer = get_optimizer(optimizer)(self._lr)
        # self.initializer = initializer
        self.valid_model = valid_model
        self.global_step = tf.Variable(0, trainable=False)
        self.decay = decay
        with tf.name_scope("Train"):
            if self.decay is not None:
                self._lr = tf.Variable(decay(0.0), dtype=tf.float32, trainable=False)
                self._new_lr = tf.placeholder(tf.float32, shape=())
                self._update_lr = tf.assign(self._lr, self._new_lr, name='update_lr')
            if gradient_clipper is None:
                self.train_op = self.optimizer(self._lr).minimize(self.model.loss, self.global_step)
            else:
                tvars = tf.trainable_variables()
                grads = gradient_clipper(tf.gradients(self.model.loss, tvars))
                # grads_and_vars = self.optimizer.compute_gradients(model.loss)
                # # Clip Gradients
                # grads, tvars = zip(*grads_and_vars)
                # grads = gradient_clipper(grads)
                self.train_op = self.optimizer.apply_gradients(
                    zip(grads, tvars),
                    global_step=self.global_step)

        self.sv = None

    def update_lr(self, sess):
        if self.decay is None:
            return
        global_step = tf.train.global_step(sess, self.global_step)
        new_lr = self.decay(global_step)
        sess.run(self._update_lr, feed_dict={self._new_lr: new_lr})

    def train(self, inputs, targets, epoch_size, epoch_num, sv, verbose=True):
        """
        Training using given input and target data
        :param inputs: a Tensor of shape [num_step, batch_size (, feature_size)] produced using data_utils.data_feeder
        :param targets: a Tensor of shape [num_step, batch_size (, feature_size)] produced using data_utils.data_feeder
        :param epoch_size: the size of one epoch
        :param epoch_num: number of training epochs
        :param valid_inputs: a Tensor of shape [1, batch_size] produced using data_utils.data_feeder
        :param valid_targets: a Tensor of shape [1, batch_size] produced using data_utils.data_feeder
        :param save_path: the path to save the logs
        :return: None
        """

        with sv.managed_session() as sess:
            for i in range(epoch_num):
                if verbose:
                    print("Epoch {}:".format(i))
                self.train_one_epoch(inputs, targets, epoch_size, sess, verbose=verbose)
                self.update_lr(sess)
            # print("Saving model to %s." % FLAGS.save_path)
            # if save_path is not None:
            #     sv.saver.save(sess, save_path, global_step=self.sv.global_step)

    def train_one_epoch(self, inputs, targets, epoch_size, sess, verbose=True):
        """
        Run one epoch of training (validating)
        :param inputs: same as above
        :param targets: same as above
        :param epoch_size: same as above
        :param run_ops: the specified TF ops to run
        :param sess: the tf.Session to run the training(validating) graph
        :param valid: if True, run validating graph
        :param verbose: flag of printing mid results
        :return: avg. results
        """
        run_ops = {'train_op': self.train_op}
        self.model.run(inputs, targets, epoch_size, run_ops, sess, verbose_every=epoch_size//10 if verbose else False)

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


def _get_optimizer(optimizer):
    """
    Simple helper to get TensorFlow Optimizer
    :param optimizer: a str specifying the name of the optimizer to use,
        or a callable function that is already a TF Optimizer
    :return: a TensorFlow Optimizer
    """
    if callable(optimizer):
        return optimizer
    if optimizer in _str2optimizers:
        return _str2optimizers[optimizer]


class Trainer(object):
    """
    Trainer Class
    """
    def __init__(self, model, optimizer, learning_rate, valid_model=None):
        """
        :param model: a instance of RNNModel class, the rnn model to be trained
        :param optimizer: the optimizer used to minimize model.loss, should be instance of tf.train.Optimizer
        :param learning_rate: the learning_rate to use
        :param supervisor: the supervisor used to manage training and saving model checkpoints
        :param valid_model: a instance of RNNModel class, the model used to run on validation set
        """
        if not isinstance(model, rnn.RNNModel):
            raise TypeError("model should be of class RNNModel!")
        self.model = model
        self.optimizer = _get_optimizer(optimizer)
        # self.initializer = initializer
        self._lr = learning_rate
        self.valid_model = valid_model
        self.train_op = self.optimizer(self._lr).minimize(self.model.loss)
        # assert isinstance(supervisor, tf.train.Supervisor)
        self.sv = None

    def train(self, inputs, targets, epoch_size, epoch_num, valid_inputs=None, valid_targets=None, save_path=None, verbose=True):
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
        self.sv = tf.train.Supervisor(logdir=save_path)
        with self.sv.managed_session() as sess:
            for i in range(epoch_num):
                if verbose:
                    print("Epoch {}:".format(i))
                self.run_one_epoch(inputs, targets, epoch_size,
                                   {'train_op': self.train_op}, sess, verbose=verbose)
                if valid_inputs is not None:
                    self.run_one_epoch(valid_inputs, valid_targets,
                                       epoch_size*self.valid_model.num_steps // self.model.num_steps,
                                       {}, sess, valid=True, verbose=verbose)
            print("Saving model to %s." % FLAGS.save_path)
            if save_path is not None:
                self.sv.saver.save(sess, save_path, global_step=self.sv.global_step)

    def run_one_epoch(self, inputs, targets, epoch_size, run_ops, sess, valid=False, verbose=True):
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
        model = self.valid_model if valid else self.model
        state = model.init_state(sess)
        run_ops['state'] = model.final_state
        run_ops['loss'] = model.loss
        total_loss = 0
        start_time = verbose_time = time.time()
        for i in range(epoch_size):
            feed_dict = model.feed_state(state)
            _inputs, _targets = sess.run([inputs, targets])
            feed_dict.update(model.feed_data(_inputs, True))
            feed_dict.update(model.feed_data(_targets, False))
            # feed_dict[model.target_holders] = [targets[:, i] for i in range(model.num_steps)]
            vals = sess.run(run_ops, feed_dict)
            state = vals['state']
            total_loss += vals['loss']
            if verbose and i % (epoch_size // 10) == 0 and i != 0:
                print("epoch[{:d}/{:d}] local avg loss:{:.3f}, speed:{:.1f} wps".format(
                    i, epoch_size, vals['loss'],
                    (epoch_size // 10)*model.batch_size*model.num_steps / (time.time()-verbose_time)))
                verbose_time = time.time()
        total_time = time.time()-start_time
        if verbose:
            print("Epoch Summary: avg loss:{:.3f}, total time:{:.1f}s, speed:{:.1f} wps".format(
                total_loss / epoch_size, total_time, epoch_size*model.num_steps*model.batch_size/total_time))
        return total_loss / epoch_size


class Evaluator(object):
    """
    An evaluator evaluates a trained model.
    This class also provides several utilities for recording hidden states
    """
    def __init__(self, model):
        self.model = model
        self._init_state = None
        self._final_state = None

    def evaluate(self, inputs, targets, size, save_path):
        pass
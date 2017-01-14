"""
Trainer handling TensorFlow Graph training computations
"""

import tensorflow as tf
from . import rnn
from .config import *


class Trainer(object):
    """
    Trainer Class
    """
    def __init__(self, model, optimizer, learning_rate, valid_model=None):
        if not isinstance(model, rnn.RNNModel):
            raise TypeError("model should be of class RNNModel!")
        self.model = model
        self.optimizer = optimizer
        # self.initializer = initializer
        self._lr = learning_rate
        self._init_state = None
        self._final_state = None
        self.train_op = optimizer(self._lr).minimize(self.model.loss)
        self.sv = None

    def train(self, inputs, targets, epoch_size, epoch_num, save_path=None):
        """
        Training using given input and target data
        :param inputs: a Tensor of shape [batch_size, num_step] produced using data_utils.data_feeder
        :param targets: a Tensor of shape [batch_size, num_step] produced using data_utils.data_feeder
        :param epoch_num: number of training epochs
        :param batch_size: batch_size
        :param validation_set: Validation set, should be {"input": input, "target": target}
            if not None, trainer will validate the model using this set after evey epoch
        :param validation_batch_size: batch_size of validation set
        :return: None
        """

        if save_path is None:
            save_path = FLAGS.save_path
        if self.sv is None:
            self.sv = tf.train.Supervisor(logdir=save_path)
        with self.sv.managed_session() as sess:
            for i in range(epoch_num):
                self.run_one_epoch(inputs, targets, epoch_size,
                                   {'loss': self.model.loss, 'train_op': self.train_op}, sess)

    def run_one_epoch(self, inputs, targets, epoch_size, run_ops, sess, verbose=False):
        model = self.model
        state = model.init_state(sess)
        run_ops['state'] = model.final_state
        total_loss = 0
        for i in range(epoch_size):
            feed_dict = model.feed_state(state)
            feed_dict[model.input_holders] = [inputs[:, i] for i in range(model.num_steps)]
            feed_dict[model.target_holders] = [targets[:, i] for i in range(model.num_steps)]
            vals = sess.run(run_ops, feed_dict)
            state = vals['state']
            total_loss += vals['loss']
            if verbose and i % (epoch_size // 10) == 0 and i != 0:
                print("epoch[{} / {}] local loss: {}".format(i, epoch_size, vals['loss']))

        if verbose:
            print("")
        return total_loss / epoch_size

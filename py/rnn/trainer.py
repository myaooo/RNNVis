"""
Trainer handling TensorFlow Graph training computations
"""

import tensorflow as tf
from . import rnn

_str2optimizers = {
    "Adam": tf.train.AdamOptimizer,
    "Adagrad": tf.train.AdagradOptimizer,
    "Momentum": tf.train.MomentumOptimizer,
    "GradientDescent": tf.train.GradientDescentOptimizer
}


def get_optimizer(optimizer, **kwargs):
    """
    Simple helper to get TensorFlow Optimizer
    :param optimizer: a str specifying the name of the optimizer to use,
        or a subclass of a TF Optimizer
    :return: a TensorFlow Optimizer
    """
    if isinstance(optimizer, str):
        if optimizer in _str2optimizers:
            optimizer = _str2optimizers[optimizer]
    try:
        if issubclass(optimizer, tf.train.Optimizer):
            pass
    except:
        raise TypeError('optimizer mal type {:s}. Should be an instance of tf.train.Optimizer or a str!'.
                        format(str(type(optimizer))))
    return lambda lr: optimizer(lr, **kwargs)

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
        return lambda t_list: tf.clip_by_global_norm(t_list, *args, **kwargs)[0]
    if clipper in _str2clipper:
        clipper = _str2clipper[clipper]
    else:
        raise ValueError('clipper should be a callable function or a given key in _str2clipper!')
    return lambda t_list: [clipper(t, *args, **kwargs) for t in t_list]


_str2decay = {
    'exponential': tf.train.exponential_decay,
    'inverse_time': tf.train.inverse_time_decay,
    'natural_exp': tf.train.natural_exp_decay,
    'polynomial': tf.train.polynomial_decay
}


def get_lr_decay(decay, *args, **kwargs):
    """
    Get a more convenient learning rate decay function.
    Note that global_step and decay_steps are neglected for convenience of using epoch_num as decay variable
    E.g. decay_func = get_lr_decay("exponential", 0.1, decay_rate=0.95)
    :param decay: a str in the keys of _str2decay, or a function of the form:
        f(epoch) -> current_learning_rate, where epoch is a Python number (float)
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
            return lambda epoch: decay_func(kwargs.pop('learning_rate'), int(epoch*10000), decay_steps=10000, **kwargs)
        else:
            return lambda epoch: decay_func(args[0], int(epoch*10000), 10000, *args[1:], **kwargs)
    else:
        raise ValueError("Cannot find corresponding decay function for the input {}".format(decay))


class Trainer(object):
    """
    Trainer Class
    Usage:
        TODO
    """
    def __init__(self, rnn_, batch_size, num_steps, keep_prob, optimizer,
                 learning_rate=0.1, gradient_clipper=None):
        """
        :param rnn_: an instance of RNN class, the rnn model to be trained
        :param batch_size: batch size of the graph
        :param num_steps: num_steps of the training graph
        :param keep_prob: keep probability of the training
        :param optimizer: the optimizer used to minimize model.loss, should be instance of tf.train.Optimizer
        :param learning_rate: a Python number denoting the learning_rate,
            or a callable function denoting the learning_rate decay function of type: f(global_step) -> current_lr
        """
        if not isinstance(rnn_, rnn.RNN):
            raise TypeError("rnn should be instance of RNN")
        self.model = rnn_.unroll(batch_size, num_steps, keep_prob, name="TrainModel")
        if callable(learning_rate):
            self.decay = learning_rate
        else:
            self._lr = tf.Variable(learning_rate, dtype=tf.float32, trainable=False)
            self.decay = None
        with tf.name_scope("Train"):
            if self.decay is not None:
                self._lr = tf.Variable(self.decay(0.0), dtype=tf.float32, trainable=False)
                self._new_lr = tf.placeholder(tf.float32, shape=())
                self._update_lr = tf.assign(self._lr, self._new_lr, name='update_lr')
            self.optimizer = optimizer(self._lr)
            # self.initializer = initializer
            self.global_step = tf.Variable(0, trainable=False)
            if gradient_clipper is None:
                self.train_op = self.optimizer.minimize(self.model.loss, self.global_step,
                                                        colocate_gradients_with_ops=True)
            else:
                grads_and_vars = self.optimizer.compute_gradients(self.model.loss, colocate_gradients_with_ops=True)
                grads, tvars = zip(*grads_and_vars)
                self.clipped_grads = gradient_clipper(grads)
                self.train_op = self.optimizer.apply_gradients(
                    list(zip(self.clipped_grads, tvars)),
                    global_step=self.global_step)
        self.current_epoch = 0

    def update_lr(self, sess, epoch_num):
        if self.decay is None:
            return
        self.current_epoch += epoch_num
        # global_step = tf.train.global_step(sess, self.global_step)
        new_lr = self.decay(self.current_epoch)
        sess.run(self._update_lr, feed_dict={self._new_lr: new_lr})

    def train(self, sess, inputs, targets, epoch_size, epoch_num, verbose=True, refresh_state=False):
        """
        Training using given input and target data
        :param inputs: a Tensor of shape [num_step, batch_size (, feature_size)] produced using data_utils.data_feeder
        :param targets: a Tensor of shape [num_step, batch_size (, feature_size)] produced using data_utils.data_feeder
        :param epoch_size: the size of one epoch
        :param epoch_num: number of training epochs
        :param sess: the session used to run the training
        :param verbose: whether print training information
        :param refresh_state: denote whether needs to re-initialize the state after each loop
        :return: None
        """

        for i in range(epoch_num):
            if verbose:
                print("Epoch:{:d}".format(i))
                print("lr:{:.3f}".format(self._lr.eval(sess)))
            self.train_one_epoch(inputs, targets, epoch_size, sess, verbose=verbose, refresh_state=refresh_state)
            self.update_lr(sess, 1)

    def train_one_epoch(self, sess, inputs, targets, epoch_size, verbose=True, refresh_state=False):
        """
        Run one epoch of training (validating)
        :param inputs: same as above
        :param targets: same as above
        :param epoch_size: same as above
        :param sess: the tf.Session to run the training(validating) graph
        :param verbose: flag of printing mid results
        :param refresh_state: denote whether needs to re-initialize the state after each loop
        :return: None
        """
        if verbose:
            print("lr:{:.3f}".format(self._lr.eval(sess)))
        run_ops = {'train_op': self.train_op}
        sum_ops = {'loss': self.model.loss}
        self.model.reset_state()
        self.model.run(inputs, targets, epoch_size, sess, run_ops,  # eval_ops={'clipped_grads': self.clipped_grads},
                       sum_ops=sum_ops, verbose=verbose, refresh_state=refresh_state)
        self.update_lr(sess, 1)

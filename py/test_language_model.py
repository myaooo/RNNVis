"""
The data required for this example is in the data/ dir of the
PTB datasets from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
"""

from py.datasets.data_utils import InputProducer
from py.datasets.ptb_reader import ptb_raw_data
from py.procedures import init_tf_environ
import py.rnn as rnn
import tensorflow as tf

flags = tf.flags
flags.DEFINE_string("config_path", None, "The path of the model configuration file")
flags.DEFINE_string("data_path", None, "The path of the input data")
flags.DEFINE_string("log_path", None, "The path to save the log")
flags.DEFINE_integer('gpu_num', 1, "The code of the gpu to use, -1 to use no gpu.")
FLAGS = flags.FLAGS


def config_path():
    return FLAGS.config_path


def data_path():
    return FLAGS.data_path


def log_path():
    return FLAGS.log_path


def test_data_producer(data, batch_size, num_steps):
    # train_data = valid_data
    producer = InputProducer(data, batch_size)
    inputs = producer.get_feeder(num_steps, transpose=True)
    targets = producer.get_feeder(num_steps, offset=1, transpose=True)
    return inputs, targets, targets.epoch_size


def test_lr_decay(epoch):
    base_lr = 1.0
    max_epoch = 4.0
    if epoch < max_epoch:
        return base_lr
    else:
        return base_lr * 0.5 ** (epoch - max_epoch)

if __name__ == '__main__':

    init_tf_environ(FLAGS.gpu_num)
    train_steps = 20
    batch_size = 20
    epoch_num = 20
    keep_prob = 1.0
    print('Preparing data')
    train_data, valid_data, test_data, vocab_size = ptb_raw_data(data_path())

    train_inputs, train_targets, epoch_size = test_data_producer(train_data, batch_size, train_steps)
    valid_inputs, valid_targets, valid_epoch_size = test_data_producer(valid_data, batch_size, train_steps)

    model = rnn.RNN('LSTM', rnn.get_initializer('random_uniform', minval=-0.1, maxval=0.1), "models/LSTM")
    model.add_cell(rnn.BasicLSTMCell, num_units=200)
    model.add_cell(rnn.BasicLSTMCell, num_units=200)
    model.set_input([None], tf.int32, 10000, 200)
    model.set_output([None, 10000], data_type())
    model.set_target([None], tf.int32)
    model.set_loss_func(rnn.get_loss_func("sequence_loss"))
    model.compile()
    model.add_trainer(batch_size, train_steps, keep_prob, rnn.trainer.get_optimizer("GradientDescent"), test_lr_decay,
                      rnn.trainer.get_gradient_clipper("global_norm", clip_norm=5.0))
    model.add_validator(batch_size, train_steps)
    print('Start Training')
    model.train(train_inputs, train_targets, epoch_size, epoch_num,
                valid_inputs=valid_inputs, valid_targets=valid_targets, valid_epoch_size=valid_epoch_size)

    print('Finish Training')
    model.save()

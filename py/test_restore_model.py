"""
Tests the restore of trained model
"""
import tensorflow as tf
from py.procedures import build_model, init_tf_environ
from py.datasets.data_utils import InputProducer
from py.datasets.ptb_reader import ptb_raw_data


flags = tf.flags
flags.DEFINE_string("config_path", None, "The path of the model configuration file")
flags.DEFINE_string("data_path", None, "The path of the input data")
flags.DEFINE_string("log_path", None, "The path to save the log")
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
    return inputs, targets, inputs.epoch_size

if __name__ == '__main__':
    init_tf_environ()
    print('Preparing data')
    train_data, valid_data, test_data, vocab_size = ptb_raw_data(data_path())

    inputs, targets, epoch_size = test_data_producer(test_data, 1, 1)

    model2 = build_model(config_path(), False)
    model2.restore()
    model2.evaluate(inputs, targets, epoch_size)

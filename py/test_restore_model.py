"""
Tests the restore of trained model
"""
import tensorflow as tf
from py.rnn.command_utils import data_path
from py.rnn.config_utils import build_rnn
from py.rnn.command_utils import config_path
from py.datasets.data_utils import InputProducer
from py.datasets.ptb_reader import ptb_raw_data


def test_data_producer(data, batch_size, num_steps):
    # train_data = valid_data
    producer = InputProducer(data, batch_size)
    inputs = producer.get_feeder(num_steps, transpose=True)
    targets = producer.get_feeder(num_steps, offset=1, transpose=True)
    return inputs, targets, inputs.epoch_size

if __name__ == '__main__':
    print('Preparing data')
    train_data, valid_data, test_data, vocab_size = ptb_raw_data(data_path())

    inputs, targets, epoch_size = test_data_producer(test_data, 1, 1)

    model2 = build_rnn(config_path())
    model2.restore()
    model2.evaluate(inputs, targets, epoch_size)

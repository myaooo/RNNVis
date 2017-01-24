"""
The data required for this example is in the data/ dir of the
PTB datasets from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
"""

from procedures import build_model
from datasets.data_utils import InputProducer
from datasets.ptb_reader import ptb_raw_data
from rnn.command_utils import config_path, data_path, init_tf_environ


def test_data_producer(data, batch_size, num_steps):
    # train_data = valid_data
    producer = InputProducer(data, batch_size)
    inputs = producer.get_feeder(num_steps, transpose=True)
    targets = producer.get_feeder(num_steps, offset=1, transpose=True)
    return inputs, targets, targets.epoch_size

if __name__ == '__main__':

    init_tf_environ()
    print('Building model..')
    model, train_config = build_model(config_path())
    train_steps = train_config.num_steps
    batch_size = train_config.batch_size
    epoch_num = train_config.epoch_num
    keep_prob = train_config.keep_prob
    print('Preparing data..')
    train_data, valid_data, test_data, vocab_size = ptb_raw_data(data_path())

    train_inputs, train_targets, epoch_size = test_data_producer(train_data, batch_size, train_steps)
    valid_inputs, valid_targets, valid_epoch_size = test_data_producer(valid_data, batch_size, train_steps)

    print('Start Training')
    model.train(train_inputs, train_targets, epoch_size, epoch_num,
                valid_inputs=valid_inputs, valid_targets=valid_targets, valid_epoch_size=valid_epoch_size)

    print('Finish Training')
    model.save()

"""
The data required for this example is in the data/ dir of the
PTB datasets from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
"""

from py.datasets.data_utils import InputProducer
from py.datasets.ptb_reader import ptb_raw_data
from py.rnn.command_utils import config_path, data_path, log_path
from py.rnn.config_utils import build_rnn, TrainConfig
from py.rnn.trainer import get_gradient_clipper


def test_data_producer(data, batch_size, num_steps):
    # train_data = valid_data
    producer = InputProducer(data, batch_size)
    inputs = producer.get_feeder(num_steps, transpose=True)
    targets = producer.get_feeder(num_steps, offset=1, transpose=True)
    return inputs, targets, targets.epoch_size


def test_lr_decay(global_step):
    base_lr = 1.0
    max_step = 1100*4
    if global_step < max_step:
        return base_lr
    else:
        return base_lr * 0.5 ** ((global_step-max_step)/1100)

if __name__ == '__main__':

    train_config = TrainConfig.load(config_path())
    logdir = log_path()
    train_steps = train_config.num_steps
    batch_size = train_config.batch_size
    epoch_num = train_config.epoch_num
    keep_prob = train_config.keep_prob
    print('Preparing data')
    train_data, valid_data, test_data, vocab_size = ptb_raw_data(data_path())

    train_inputs, train_targets, epoch_size = test_data_producer(train_data, batch_size, train_steps)
    valid_inputs, valid_targets, valid_epoch_size = test_data_producer(valid_data, batch_size, train_steps)

    model = build_rnn(config_path())
    model.add_trainer(batch_size, train_steps, keep_prob, train_config.optimizer, test_lr_decay,
                      clipper=train_config.clipper)
    model.add_validator(batch_size, train_steps)
    print('Start Training')
    model.train(train_inputs, train_targets, epoch_size, epoch_num,
                valid_inputs=valid_inputs, valid_targets=valid_targets, valid_epoch_size=valid_epoch_size)

    print('Finish Training')
    model.save()

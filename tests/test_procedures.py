"""
The data required for this example is in the data/ dir of the
PTB datasets from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
"""

from rnnvis.procedures import build_model, init_tf_environ, produce_ptb_data
from rnnvis.datasets.data_utils import InputProducer
from tensorflow import flags

flags.DEFINE_string("config_path", None, "The path of the model configuration file")
flags.DEFINE_string("data_path", None, "The path of the input data")
flags.DEFINE_string("log_path", None, "The path to save the log")
flags.DEFINE_integer('gpu_num', 0, "The number of the gpu to use, 0 to use no gpu.")
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
    inputs = producer.get_feeder(num_steps)
    targets = producer.get_feeder(num_steps, offset=1)
    return inputs, targets, targets.epoch_size

if __name__ == '__main__':

    init_tf_environ(FLAGS.gpu_num)
    print('Building model..')
    model, train_config = build_model(config_path())

    epoch_num = train_config.epoch_num
    keep_prob = train_config.keep_prob

    print('Preparing data..')
    data_producers = produce_ptb_data(data_path(), train_config, valid=True, test=True)
    train_inputs, train_targets, epoch_size = data_producers[0]
    valid_inputs, valid_targets, valid_epoch_size = data_producers[1]

    print('Start Training')
    model.train(train_inputs, train_targets, epoch_size, epoch_num,
                valid_inputs=valid_inputs, valid_targets=valid_targets, valid_epoch_size=valid_epoch_size)

    print('Finish Training')
    model.save()


"""
The data required for this example is in the data/ dir of the
PTB datasets from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
"""

from tensorflow import flags

from py.procedures import build_model, init_tf_environ
from py.datasets.data_utils import get_lm_data_producer
from py.db.language_model import get_datasets_by_name
from py.rnn.evaluator import Recorder


flags.DEFINE_string("config_path", None, "The path of the model configuration file")
flags.DEFINE_string("data_name", None, "The name of the datasets in db")
flags.DEFINE_integer('gpu_num', 0, "The number of the gpu to use, 0 to use no gpu.")
FLAGS = flags.FLAGS


def config_path():
    return FLAGS.config_path


if __name__ == '__main__':

    init_tf_environ(FLAGS.gpu_num)
    print('Building model..')
    model, train_config = build_model(config_path())
    model.add_evaluator(train_config.batch_size, 1, True, True, True, True)
    epoch_num = train_config.epoch_num
    keep_prob = train_config.keep_prob
    batch_size = train_config.batch_size
    num_steps = train_config.num_steps

    print('Preparing data..')
    datasets = get_datasets_by_name(str(FLAGS.data_name), ['train', 'valid', 'test'])
    train = datasets['train']
    valid = datasets['valid']
    test = datasets['test']
    train_inputs, train_targets, epoch_size = get_lm_data_producer(train, batch_size, num_steps)
    valid_inputs, valid_targets, valid_epoch_size = get_lm_data_producer(valid, batch_size, num_steps)
    test_inputs, test_targets, test_epoch_size = get_lm_data_producer(test, 1, 1)

    print('Start Training')
    model.train(train_inputs, train_targets, epoch_size, epoch_num,
                valid_inputs=valid_inputs, valid_targets=valid_targets, valid_epoch_size=valid_epoch_size)

    print('Finish Training')
    model.save()
    print('Testing...')
    model.validate(test_inputs, test_targets, test_epoch_size)

    model.run_with_context(model.evaluator.evaluate_and_record, [test[:1000]], [test[1:1001]],
                           Recorder('ptb', model.name), verbose=True)

"""
Tests the restore of trained model
"""
import tensorflow as tf
from py.procedures import build_model, init_tf_environ, pour_data
from py.rnn.evaluator import Recorder
from py.db import get_dataset
from py.datasets.data_utils import SentenceProducer


flags = tf.flags
flags.DEFINE_string("config_path", None, "The path of the model configuration file")
flags.DEFINE_integer('gpu_num', 0, "The number of the gpu to use, 0 to use no gpu.")
FLAGS = flags.FLAGS


def config_path():
    return FLAGS.config_path


if __name__ == '__main__':
    init_tf_environ(FLAGS.gpu_num)
    # datasets = get_datasets_by_name('ptb', ['test'])
    # test_data = datasets['test']

    model, train_config = build_model(config_path(), True)
    model.add_evaluator(1, 56, log_gradients=True)

    print('Preparing data')
    datasets = get_dataset(train_config.dataset, ['test'])
    test_data = datasets['test']
    model.restore()

    input_producer = SentenceProducer(test_data['data'][:100], 1, 56)

    model.run_with_context(model.evaluator.evaluate_and_record,
                           Recorder('ptb', model.name), verbose=True)

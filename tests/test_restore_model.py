"""
Tests the restore of trained model
"""
import tensorflow as tf
from py.procedures import build_model, init_tf_environ, pour_data
from py.rnn.evaluator import StateRecorder
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
    model.add_evaluator(20, 1, 1, True, True, False, False)

    print('Preparing data')
    producers = pour_data(train_config.dataset, ['train'], 20, 1)
    inputs, targets, epoch_size = producers[0]
    model.restore()

    model.run_with_context(model.evaluator.evaluate_and_record, inputs, targets,
                           StateRecorder(train_config.dataset, model.name), verbose=True)

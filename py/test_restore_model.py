"""
Tests the restore of trained model
"""
import tensorflow as tf
from py.procedures import build_model, init_tf_environ
from py.rnn.evaluator import Recorder
from py.db.language_model import get_datasets_by_name


flags = tf.flags
flags.DEFINE_string("config_path", None, "The path of the model configuration file")
flags.DEFINE_integer('gpu_num', 0, "The number of the gpu to use, 0 to use no gpu.")
FLAGS = flags.FLAGS


def config_path():
    return FLAGS.config_path


if __name__ == '__main__':
    init_tf_environ(FLAGS.gpu_num)
    print('Preparing data')
    datasets = get_datasets_by_name('ptb', ['test'])
    test_data = datasets['test']

    model2 = build_model(config_path(), False)
    model2.restore()
    model2.run_with_context(model2.evaluator.evaluate_and_record,
                            [test_data[:1000]], [test_data[1:1001]],
                            Recorder('ptb', model2.name), verbose=True)

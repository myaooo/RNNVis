"""
Tests: Restore a trained model and re-evaluate
"""
import tensorflow as tf
from rnnvis.procedures import build_model, init_tf_environ, pour_data

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

    model.add_evaluator(10, 1, 1, True, False, False, False, log_gates=False)
    model.restore()

    print('Preparing data')
    producers = pour_data(train_config.dataset, ['test'], train_config.batch_size, train_config.num_steps)
    inputs, targets, epoch_size = producers[0]
    print('Re-Testing...')
    model.validate(inputs, targets, epoch_size)

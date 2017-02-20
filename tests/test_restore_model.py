"""
Tests the restore of trained model
"""
import tensorflow as tf
from rnnvis.procedures import build_model, init_tf_environ, pour_data
from rnnvis.rnn.evaluator import StateRecorder
from rnnvis.db import get_dataset
from rnnvis.datasets.data_utils import SentenceProducer


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

    model.add_evaluator(1, 1, 1, True, True, False, False, log_gates=False, cal_salience=True)
    model.restore()

    # scripts that eval and record states
    if False:
        print('Preparing data')
        producers = pour_data(train_config.dataset, ['test'], 10, 1)
        inputs, targets, epoch_size = producers[0]
        model.run_with_context(model.evaluator.evaluate_and_record, inputs, targets,
                               StateRecorder(train_config.dataset, model.name, 500), verbose=True)

    salience = model.run_with_context(model.evaluator.cal_saliency, 10)


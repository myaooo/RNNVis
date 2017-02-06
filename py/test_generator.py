"""
Tests the generator
"""

import os
import tensorflow as tf
from py.procedures import build_model, init_tf_environ
from py.datasets.data_utils import load_data_as_ids
from py.utils.io_utils import lists2csv


flags = tf.flags
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


if __name__ == '__main__':

    init_tf_environ(FLAGS.gpu_num)

    model, train_config = build_model(config_path())
    _, word_to_id, id_to_word = load_data_as_ids([os.path.join(data_path(), "ptb.train.txt")])
    lists2csv([[s, v] for s, v in word_to_id.items()], os.path.join(data_path(), 'word_to_id.csv'), " ")
    model.add_generator(word_to_id)
    model.restore()
    model.generate(['once', 'again'], 'test.json', max_branch=3, accum_cond_prob=0.9,
                   min_cond_prob=0.0, min_prob=1e-10, max_step=6)



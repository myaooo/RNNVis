"""
Tests the generator
"""

import os
import tensorflow as tf
from py.procedures import build_model, init_tf_environ
from py.db.language_model import get_datasets_by_name


flags = tf.flags
flags.DEFINE_string("config_path", None, "The path of the model configuration file")
# flags.DEFINE_string("data_name", None, "The name of the datasets in db")
# flags.DEFINE_string("log_path", None, "The path to save the log")
flags.DEFINE_integer('gpu_num', 0, "The number of the gpu to use, 0 to use no gpu.")
FLAGS = flags.FLAGS


if __name__ == '__main__':

    init_tf_environ(FLAGS.gpu_num)

    model, train_config = build_model(FLAGS.config_path)
    word_to_id = get_datasets_by_name(train_config.dataset, ['word_to_id'])['word_to_id']
    # _, word_to_id, id_to_word = load_data_as_ids([os.path.join(data_path(), "ptb.train.txt")])
    # lists2csv([[s, v] for s, v in word_to_id.items()], os.path.join(data_path(), 'word_to_id.csv'), " ")
    model.add_generator(word_to_id)
    model.restore()
    model.generate(['the', 'meaning', 'of', 'life', 'is'], 'test.json', max_branch=3, accum_cond_prob=0.9,
                   min_cond_prob=0.005, min_prob=1e-15, max_step=20, neg_word_ids=['<unk>', 'N', '<eos>', '$'])



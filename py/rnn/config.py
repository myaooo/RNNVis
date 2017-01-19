"""
Configurations for RNN models
"""

import yaml
import tensorflow as tf

tf.GraphKeys.EVAL_SUMMARIES = "eval_summarys"

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


# class RNNConfig(object):
#     """
#     A helper class that specify the configuration of RNN models
#     """
#     def __init__(self, filename):
#         self.filename = filename
#         f = open(filename)
#         self._dict = yaml.safe_load(f)
#         f.close()
#         self.__dict__.update(self._dict)
#
#     def save2yaml(self, filename):
#         f = open(filename, 'w')
#         yaml.dump(self._dict, f)
#         f.close()
#

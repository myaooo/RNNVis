"""
Configurations for RNN models
"""

import os
import tensorflow as tf
# import gflags as flags
# from tensorflow.python.client import device_lib
#
#
# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']

flags = tf.flags
flags.DEFINE_bool("use_fp16", False, "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_float('gpu_memory', 0.5, "The fraction of gpu memory each process is allowed to use")
flags.DEFINE_integer('gpu_num', 1, "The number of gpu to use.")
flags.DEFINE_string("config_path", None, "The path of the model configuration file")
flags.DEFINE_string("data_path", None, "The path of the input data")
flags.DEFINE_string("log_path", None, "The path to save the log")


# class FLAGS:
#     use_fp16 = False
#     config_path = None
#     data_path = None
#     log_path = None
#     gpu_memory = 0.5
#     gpu_num = 1

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


def config_path():
    return FLAGS.config_path


def data_path():
    return FLAGS.data_path


def log_path():
    return FLAGS.log_path


def config_proto():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory)
    return tf.ConfigProto(device_count={"GPU": FLAGS.gpu_num}, gpu_options=gpu_options)


def init_tf_environ():
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if FLAGS.gpu_num == 0 else "0,1,2,3"[:(FLAGS.gpu_num * 2 - 1)]

"""
Configurations for RNN models
"""

import subprocess
import re

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


def config_proto():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory)
    return tf.ConfigProto(device_count={"GPU": 1}, gpu_options=gpu_options)


# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")


def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse "+line
        result.append(int(m.group("gpu_id")))
    return result


def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result


def pick_gpu_lowest_memory(num=1):
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memorys, best_gpus = sorted(memory_gpu_map)[:num]
    return best_gpus

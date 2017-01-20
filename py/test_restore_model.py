"""
Tests the restore of trained model
"""
import tensorflow as tf
from py.rnn.config_utils import build_rnn
from py.rnn.command_utils import config_path

if __name__ == '__main__':
    model2 = build_rnn(config_path())
    model2.restore()

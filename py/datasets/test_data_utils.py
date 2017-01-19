"""
Tests for data utils
"""

import tensorflow as tf
from py.datasets.data_utils import data_batcher, data_feeder


def test_data_feeder():
    raw_data = list(range(10000))
    batched_data = data_batcher(raw_data, 20)
    inputs = data_feeder(batched_data, 20)
    sv = tf.train.Supervisor()
    outputs = []
    with sv.managed_session() as sess:
        sess.run(inputs)
        outputs.append(sess.run(inputs))
    print(outputs)


if __name__ == '__main__':
    test_data_feeder()

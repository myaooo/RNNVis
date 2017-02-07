"""
Tests for data utils
"""

import tensorflow as tf
from py.datasets.data_utils import InputProducer


def test_data_feeder():
    raw_data = list(range(10000))
    producer = InputProducer(raw_data, 20)
    feeder = producer.get_feeder(20)
    print(feeder())
    print(feeder())

if __name__ == '__main__':
    test_data_feeder()

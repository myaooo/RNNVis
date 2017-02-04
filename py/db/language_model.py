"""
Use MongoDB to manage all the datasets
"""

import os

import numpy as np

from py.utils.io_utils import dict2json, get_path
from py.datasets.data_utils import load_data_as_ids, split
from py.datasets.text_processor import PlainTextProcessor
from py.db import mongo


db_name = 'language_model'
collections = {
    'word_to_id': {'name': str, 'data': str},
    'train': {'name': str, 'data': list},
    'valid': {'name': str, 'data': list},
    'test': {'name': str, 'data': list}
}

db_hdlr = mongo[db_name]


def store_ptb(data_path, name='ptb', upsert=True):
    """
    Process and store the ptb datasets to db
    :param data_path:
    :param name:
    :return:
    """
    train_path = get_path(data_path, "ptb.train.txt")
    valid_path = get_path(data_path, "ptb.valid.txt")
    test_path = get_path(data_path, "ptb.test.txt")
    paths = [train_path, valid_path, test_path]

    data_list, word_to_id = load_data_as_ids(paths)
    train, valid, test = data_list
    word_to_id_json = dict2json(word_to_id)
    if upsert:
        insertion = _replace_one_if_exists
    else:
        insertion = _insert_one_if_not_exists

    insertion('word_to_id', {'name': name}, {'name': name, 'data': word_to_id_json})
    insertion('train', {'name': name}, {'name': name, 'data': train})
    insertion('valid', {'name': name}, {'name': name, 'data': valid})
    insertion('valid', {'name': name}, {'name': name, 'data': test})


def store_plain_text(data_path, name, split_scheme, min_freq=1, max_vocab=10000, upsert=True):
    """
    Process any plain text and store to db
    :param data_path:
    :param name:
    :param split_scheme:
    :param min_freq:
    :param max_vocab:
    :param replace:
    :return:
    """
    if upsert:
        insertion = _replace_one_if_exists
    else:
        insertion = _insert_one_if_not_exists
    processor = PlainTextProcessor(get_path(data_path))
    processor.tag_rare_word(min_freq, max_vocab)
    split_ids = split(processor.flat_ids, split_scheme.values())
    split_data = dict(zip(split_scheme.keys(), split_ids))
    if 'train' not in split_data:
        print('WARN: there is no train data in the split data!')
    for c_name in ['train', 'valid', 'test']:
        if c_name in split_data:
            insertion(c_name, {'name': name}, {'name': name, 'data': split_data[c_name]})
    insertion('word_to_id', {'name': name}, {'name': name, 'data': dict2json(processor.word_to_id)})


def _insert_one_if_not_exists(c_name, filter_, data):
    if db_hdlr[c_name].find_one(filter_) is not None:
        raise ValueError('the data of signature {:s} is already exists in collection {:s}'.format(filter_, c_name))
    return db_hdlr[c_name].insert_one(data)


def _replace_one_if_exists(c_name, filter_, data):
    results = db_hdlr[c_name].replace_one(filter_, data, upsert=True)
    if results.upserted_id is not None:
        print('WARN: a document with signature {:s} in the collection {:s} of db {:s} has been replaced'.
              format(filter_, c_name, db_name))
    return results

if __name__ == '__main__':
    store_ptb('cached_data/simple-examples/data')
    store_plain_text('cached_data/tinyshakespeare.txt')

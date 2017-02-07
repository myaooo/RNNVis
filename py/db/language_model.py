"""
Use MongoDB to manage all the language modeling datasets
"""

import os
import json
import pickle
import hashlib

import yaml
import numpy as np
from pymongo import ReturnDocument
from bson.binary import Binary
from bson.objectid import ObjectId

from py.utils.io_utils import dict2json, get_path
from py.datasets.data_utils import load_data_as_ids, split
from py.datasets.text_processor import PlainTextProcessor
from py.db import mongo

db_name = 'language_model'
# db definition
collections = {
    'word_to_id': {'name': (str, 'unique', 'name of the dataset'),
                   'data': (str, '', 'a json str, encoding word to id mapping')},
    'id_to_word': {'name': (str, 'unique', 'name of the dataset'),
                   'data': (list, '', 'a list of str')},
    'train': {'name': (str, 'unique', 'name of the dataset'),
              'data': (list, '', 'a list of word_ids (int), train data')},
    'valid': {'name': (str, 'unique', 'name of the dataset'),
              'data': (list, '', 'a list of word_ids (int), valid data')},
    'test': {'name': (str, 'unique', 'name of the dataset'),
             'data': (list, '', 'a list of word_ids (int), test data')},
    'eval': {'name': (str, '', 'name of the dataset'),
             'tag': (str, 'unique', 'a unique tag of hash of the evaluating text sequence'),
             'data': (list, '', 'a list of word_ids (int), eval text'),
             'model': (str, '', 'the identifier of the model that the sequence evaluated on'),
             'records': (list, '', 'a list of ObjectIds of the records')},
    'record': {'word_id': (int, '', 'word_id in the word_to_id values'),
               'state': (np.ndarray, 'optional', 'hidden state np.ndarray'),
               'state_c': (np.ndarray, 'optional', 'hidden state np.ndarray'),
               'state_h': (np.ndarray, 'optional', 'hidden state np.ndarray'),
               'output': (np.ndarray, '', 'raw output (unprojected)')},
    'model': {'name': (str, '', 'identifier of a trained model'),
              'data_name': (str, '', 'name of the datasets that the model uses')}
}

db_hdlr = mongo[db_name]


def get_datasets_by_name(name, fields=None):
    complete_data = {}
    fields = ['word_to_id', 'id_to_word', 'train', 'valid', 'test'] if fields is None else fields
    for c_name in fields:
        results = db_hdlr[c_name].find_one({'name': name})
        if results is None:
            print('WARN: No data in collection {:s} of db {:s} named {:s}'.format(c_name, db_name, name))
            return None
        complete_data[c_name] = results['data']
    if 'word_to_id' in complete_data:
        complete_data['word_to_id'] = json.loads(complete_data['word_to_id'])
    return complete_data


def store_ptb(data_path, name='ptb', upsert=False):
    """
    Process and store the ptb datasets to db
    :param data_path:
    :param name:
    :param upsert:
    :return:
    """
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
    paths = [train_path, valid_path, test_path]

    data_list, word_to_id, id_to_word = load_data_as_ids(paths)
    train, valid, test = data_list
    word_to_id_json = dict2json(word_to_id)
    if upsert:
        insertion = _replace_one_if_exists
    else:
        insertion = _insert_one_if_not_exists

    insertion('word_to_id', {'name': name}, {'name': name, 'data': word_to_id_json})
    insertion('id_to_word', {'name': name}, {'name': name, 'data': id_to_word})
    insertion('train', {'name': name}, {'name': name, 'data': train})
    insertion('valid', {'name': name}, {'name': name, 'data': valid})
    insertion('test', {'name': name}, {'name': name, 'data': test})


def store_plain_text(data_path, name, split_scheme, min_freq=1, max_vocab=10000, upsert=False):
    """
    Process any plain text and store to db
    :param data_path:
    :param name:
    :param split_scheme:
    :param min_freq:
    :param max_vocab:
    :param upsert:
    :return:
    """
    if upsert:
        insertion = _replace_one_if_exists
    else:
        insertion = _insert_one_if_not_exists
    processor = PlainTextProcessor(data_path)
    processor.tag_rare_word(min_freq, max_vocab)
    split_ids = split(processor.flat_ids, split_scheme.values())
    split_data = dict(zip(split_scheme.keys(), split_ids))
    if 'train' not in split_data:
        print('WARN: there is no train data in the split data!')
    for c_name in ['train', 'valid', 'test']:
        if c_name in split_data:
            insertion(c_name, {'name': name}, {'name': name, 'data': split_data[c_name]})
    insertion('word_to_id', {'name': name}, {'name': name, 'data': dict2json(processor.word_to_id)})
    insertion('id_to_word', {'name': name}, {'name': name, 'data': processor.id_to_word})


def _insert_one_if_not_exists(c_name, filter_, data):
    results = db_hdlr[c_name].find_one(filter_)
    if results is not None:
        print('The data of signature {:s} is already exists in collection {:s}.\n Pass.'.format(str(filter_), c_name))
        return results
    return db_hdlr[c_name].insert_one(data)


def _replace_one_if_exists(c_name, filter_, data):
    results = db_hdlr[c_name].find_one_and_replace(filter_, data, upsert=True, return_document=ReturnDocument.AFTER)
    if results is not None:
        print('WARN: a document with signature {:s} in the collection {:s} of db {:s} has been replaced'
              .format(str(filter_), c_name, db_name))
    return results


def seed_db():
    """
    Use the `config/datasets/lm.yml` to generate example datasets and store them into db.
    :return: None
    """
    config_dir = get_path('config/db', 'lm.yml')
    with open(config_dir, 'r') as f:
        config = yaml.safe_load(f)['datasets']
    for seed in config:
        data_dir = get_path('cached_data', seed['dir'])
        print('seeding {:s} data'.format(seed['name']))
        if seed['type'] == 'ptb':
            store_ptb(data_dir, seed['name'])
        elif seed['type'] == 'text':
            store_plain_text(data_dir, seed['name'], **seed['scheme'])
        else:
            print('cannot find corresponding seed functions')


def insert_evaluation(data_name, model_name, eval_text_tokens):
    """
    Add an evaluation record to the 'eval' collection, with given data_name, model_name and a list of word tokens
    :param data_name: name of the datasets, should be in word_to_id collection
    :param model_name: name of the model, identifier
    :param eval_text_tokens: a list of word tokens, or a list of word_ids
    :return: the ObjectId of the inserted evaluation document
    """
    assert isinstance(eval_text_tokens, list)
    if isinstance(eval_text_tokens[0], int):
        id_to_word = get_datasets_by_name(data_name, ['id_to_word'])['id_to_word']
        eval_ids = eval_text_tokens
        try:
            eval_text_tokens = [id_to_word[i] for i in eval_ids]
        except:
            print('word id of input eval text and dictionary not match!')
            raise
    elif isinstance(eval_text_tokens[0], str):
        word_to_id = get_datasets_by_name(data_name, ['word_to_id'])['word_to_id']
        try:
            eval_ids = [word_to_id[token] for token in eval_text_tokens]
        except:
            print('token and dictionary not match!')
            raise
    else:
        raise TypeError('The input eval text should be a list of int or a list of str! But its of type {}'
                        .format(type(eval_text_tokens[0])))
    tag = hash_tag_str(eval_text_tokens)
    filt = {'name': data_name, 'tag': tag, 'model': model_name}
    data = {'data': eval_ids}
    data.update(filt)
    doc = _replace_one_if_exists('eval', filt, data)
    return doc['_id']


def push_evaluation_records(eval_ids, records):
    """
    Add a detailed record (of one step) of evaluation into db, and update the corresponding doc in 'eval'
    :param eval_ids: a list, with each element as the ObjectId returned by insert_evaluation()
    :param records: a list, with each element as a dict of record to be inserted,
        record and eval_id must match for each element
    :return: a pair of results (insert_many_result, update_result)
    """
    # check
    for record in records:
        if 'word_id' not in record:
            raise KeyError('there is no key named "word_id" in record!')
        # convert np.ndarry into binary
        for key, value in record.items():
            if isinstance(value, np.ndarray):
                record[key] = Binary(pickle.dumps(value, protocol=3))
            elif isinstance(value, str) or isinstance(value, int):
                pass
            else:
                print('Unkown type {:s}'.format(str(type(value))))

    results = db_hdlr['record'].insert_many(records)
    update_results = []
    for i, eval_id in enumerate(eval_ids):
        update_results.append(db_hdlr['eval'].update_one({'_id': eval_id},
                                                         {'$push': {'records': results.inserted_ids[i]}}))
    return results, update_results


def query_evaluation_records(eval_, range_, data_name=None, model_name=None):
    """
    Query for the evaluation records
    :param eval_: a eval_id of type ObjectId, or a list of tokens, or a hash tag of the tokens
    :param range_: a range object, specifying the range of indices of records
    :param data_name: optional, when `eval` is not a ObjectId, this field should be filled
    :param model_name: optional, when `eval` is not a ObjectId, this field should be filled
    :return: a list of records
    """
    if isinstance(eval_, ObjectId):
        # ignoring data_name and model_name
        eval_record = db_hdlr['eval'].find_one({'_id': eval_})
    else:
        if isinstance(eval_, list):
            try:
                tag = hash_tag_str(eval_)
            except:
                print("Unable to hash the eval_ list!")
                raise
        elif isinstance(eval_, str):
            tag = eval_
        else:
            raise TypeError("Expecting type ObjectId, list or str, but receive type {:s}".format(str(type(eval_))))
        eval_record = db_hdlr['eval'].find_one({'tag':tag, 'name': data_name, 'model': model_name})
    ids = eval_record['records']
    records = []
    for i in range_:
        record = db_hdlr['record'].find_one({'_id': ids[i]})
        for name, value in record:
            if isinstance(value, Binary):
                record[name] = pickle.loads(value)
        records.append(record)
    return records


def hash_tag_str(text_list):
    """Use hashlib.md5 to tag a hash str of a list of text"""
    return hashlib.md5(" ".join(text_list).encode()).hexdigest()


if __name__ == '__main__':
    # store_ptb(get_path('cached_data/simple-examples/data'))
    # store_plain_text(get_path('cached_data/tinyshakespeare.txt'), 'shakespeare', {'train': 0.9, 'valid': 0.05, 'test': 0.05})
    seed_db()

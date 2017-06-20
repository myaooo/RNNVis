"""
Common helper functions that db package will use
"""

import os
import json
import pickle
import hashlib

import numpy as np
from pymongo import ReturnDocument
from bson.binary import Binary
from bson.objectid import ObjectId

from rnnvis.db import mongo
from rnnvis.utils.io_utils import get_path, dict2json, file_exists, path_exists

_db_name = 'rnnvis'
datasets_path = '_cached/datasets'

collections = {
    'datasets': {'name': (str, 'unique', 'name of the dataset'),
                 'type': (str, '', 'type of the dataset, currently support sp and lm')},
    'word_to_id': {'name': (str, 'unique', 'name of the dataset'),
                   'data': (str, '', 'a json str, encoding word to id mapping')},
    'id_to_word': {'name': (str, 'unique', 'name of the dataset'),
                   'data': (list, '', 'a list of str')},
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

db_hdlr = mongo[_db_name]


def db_handler(db_name):
    return mongo[db_name]


# def insert_one_if_not_exists(db_name, c_name, filter_, data):
#     results = db_handler(db_name)[c_name].find_one(filter_)
#     if results is not None:
#         print('The data of signature {:s} is already exists in collection {:s}.\n Pass.'.format(str(filter_), c_name))
#         return results
#     return db_handler(db_name)[c_name].insert_one(data)


def insert_many_if_not_exists(db_name, c_name, filter_, data):
    results = db_handler(db_name)[c_name].find_one(filter_)
    if results is not None:
        print('The data of signature {:s} is already exists in collection {:s}.\n Pass.'.format(str(filter_), c_name))
        return results
    return db_handler(db_name)[c_name].insert_many(data)


# def replace_one_if_exists(db_name, c_name, filter_, data):
#     results = db_handler(db_name)[c_name].find_one_and_replace(filter_, data, upsert=True,
#                                                                return_document=ReturnDocument.AFTER)
#     if results is not None:
#         print('WARN: a document with signature {:s} in the collection {:s} of db {:s} has been replaced'
#               .format(str(filter_), c_name, db_name))
#     return results


def replace_all_if_exists(db_name, c_name, filter_, data):
    delete_result = delete_many(db_name, c_name, filter_)
    results = db_handler(db_name)[c_name].insert_many(data)
    if delete_result.deleted_count > 0:
        print('''WARN: {:d} documents matching signature {:s} in the collection {:s} of db {:s} has been deleted, \
                 and {:d} documents has been inserted
              '''
              .format(delete_result.deleted_count, str(filter_), c_name, db_name, len(results.inserted_ids)))
    return results


def insert_one_if_not_exists(c_name, filter_, data):
    results = db_hdlr[c_name].find_one(filter_)
    if results is not None:
        print('The data of signature {:s} is already exists in collection {:s}.\n Pass.'.format(str(filter_), c_name))
        return results
    return db_hdlr[c_name].insert_one(data)


def replace_one_if_exists(c_name, filter_, data):
    find_cusor = db_hdlr[c_name].find(filter_)
    if find_cusor.count() != 0:
        print('WARN: a document with signature {:s} in the collection {:s} of db {:s} has been replaced'
              .format(str(filter_), c_name, _db_name))
    results = db_hdlr[c_name].find_one_and_replace(filter_, data, upsert=True, return_document=ReturnDocument.AFTER)
    return results


def delete_many(db_name, c_name, filter_):
    print("WARN: Deleting data matching signature {:s} from collection: {:s} of db: {:s}".
          format(str(filter_), c_name, db_name))
    return db_handler(db_name)[c_name].delete_many(filter_)


def dataset_inserted(name, data_type, force=False):
    # assert data_type == 'lm' or data_type == 'sp', "Unkown type {:s}".format(str(data_type))
    exists = db_hdlr['datasets'].find({'name': name})
    if exists.count() == 0:
        return db_hdlr['datasets'].insert_one({'name': name, 'type': data_type})
    elif force:
        return db_hdlr['datasets'].replace_one({'name': name}, {'name': name, 'type': data_type})
    else:
        print("datasets record for {:s} already exists. Use force if you want to overwrite".format(name))
        return None


def get_dataset_path(name):
    return get_path(datasets_path, name)


def store_dataset_by_default(name, data_dict, force=False):
    dataset_path = get_dataset_path(name)
    for field, data in data_dict.items():
        target_path = os.path.join(dataset_path, field)
        if file_exists(target_path):
            if force:
                dict2json(data, target_path)
                print("{:s} data already exists, overwritten.".format(field))
            else:
                print("{:s} data already exists, if you want to overwrite, use force!".format(field))
        else:
            dict2json(data, target_path)


def get_datasets_by_name(name, fields=None):
    complete_data = {}
    fields = ['word_to_id', 'id_to_word', 'train', 'valid', 'test'] if fields is None else fields
    for c_name in fields:
        if c_name in ['train', 'test', 'valid']:
            data = json.load(open(get_path(get_dataset_path(name), c_name)))
            complete_data[c_name] = data
            continue
        results = db_hdlr[c_name].find_one({'name': name})
        if results is None:
            print('WARN: No data in collection {:s} of db {:s} named {:s}'.format(c_name, _db_name, name))
            return None
        complete_data[c_name] = results['data']
    if 'word_to_id' in complete_data:
        complete_data['word_to_id'] = json.loads(complete_data['word_to_id'])
    return complete_data


def insert_evaluation(data_name, model_name, set_name, eval_text_tokens, replace=False):
    """
    Add an evaluation record to the 'eval' collection, with given data_name, model_name and a list of word tokens
    Since we don't want duplicate evaluations, we must ensure one time there is only one same doc
    :param data_name: name of the datasets, should be in word_to_id collection
    :param model_name: name of the model, identifier
    :param eval_text_tokens: a list of eval_sentences, each eval_sentence is a list of tokens,
        or a single eval_sentence is also acceptable
    :param force: if True, this will replace an already exists evaluation document, if exists
        if False, then when encountered a different documents, it will not overwrite it.
    :return: a list of ObjectIds of the inserted evaluation documents
    """
    assert isinstance(eval_text_tokens, list)
    if not isinstance(eval_text_tokens[0], list): # convert a single sentence to standard list
        eval_text_tokens = [eval_text_tokens]
    dictionaries = get_datasets_by_name(data_name, ['id_to_word', 'word_to_id'])
    id_to_word = dictionaries['id_to_word']
    word_to_id = dictionaries['word_to_id']
    doc_ids = []
    datas = []
    for eval_text_token in eval_text_tokens:
        if isinstance(eval_text_token[0], int):
            eval_ids = eval_text_token
            try:
                eval_text_token = [id_to_word[i] for i in eval_ids]
            except:
                print('word id of input eval text and dictionary not match!')
                raise
        elif isinstance(eval_text_token[0], str):
            try:
                eval_ids = [word_to_id[token] for token in eval_text_token]
            except:
                print('token and dictionary not match!')
                raise
        else:
            raise TypeError('The input eval text should be a list of int or a list of str! But its of type {}'
                            .format(type(eval_text_token[0])))
        tag = hash_tag_str(eval_text_token)
        filt = {'name': data_name, 'tag': tag, 'model': model_name}
        if set_name is not None:
            filt['set'] = set_name
        data = {'data': eval_ids}
        data.update(filt)
        datas.append(data)
    if replace:
        existing_evals = query_evals(data_name, model_name, set_name)
        delete_evals([eval_['_id'] for eval_ in existing_evals])
        return db_hdlr['eval'].insert_many(datas).inserted_ids
    else:
        return db_hdlr['eval'].insert_many(datas).inserted_ids
    # return doc_ids


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


def query_evaluation_records(eval_, range_=None, data_name=None, model_name=None):
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
    range_ = range(len(ids)) if range_ is None else range_
    for i in range_:
        record = db_hdlr['record'].find_one({'_id': ids[i]})
        for name, value in record.items():
            if isinstance(value, bytes):
                record[name] = pickle.loads(value)
        records.append(record)
    return records


def query_evals(data_name, model_name, set_name=None):
    filt = {'name': data_name, 'model': model_name}
    if set_name is not None:
        filt['set'] = set_name
    return db_hdlr['eval'].find(filt)


def delete_evals(eval_ids):
    if isinstance(eval_ids, ObjectId):
        eval_ids = [eval_ids]
    evals = db_hdlr['eval'].find({'_id': {'$in': eval_ids}})
    expect_record_num = 0
    deleted_record_num = 0
    for eval_ in evals:
        if 'records' not in eval_:
            continue
        record_ids = eval_['records']
        del_results = db_hdlr['record'].delete_many({'_id': {'$in': record_ids}})
        expect_record_num += len(record_ids)
        deleted_record_num += del_results.deleted_count
        if del_results.deleted_count != len(record_ids):
            print("WARN: Deleted records fewer than the records in the eval doc!")
    del_evals = db_hdlr['eval'].delete_many({'_id': {'$in': eval_ids}})
    print("{:d} eval docs are deleted, with {:d} out of {:d} expected record docs deleted"
          .format(del_evals.deleted_count, deleted_record_num, expect_record_num))
    return expect_record_num == deleted_record_num


def hash_tag_str(text_list):
    """Use hashlib.md5 to tag a hash str of a list of text"""
    return hashlib.md5(" ".join(text_list).encode()).hexdigest()

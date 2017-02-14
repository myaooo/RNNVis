"""
Common helper functions that db package will use
"""

from pymongo import ReturnDocument

from rnnvis.db import mongo


def db_handler(db_name):
    return mongo[db_name]


def insert_one_if_not_exists(db_name, c_name, filter_, data):
    results = db_handler(db_name)[c_name].find_one(filter_)
    if results is not None:
        print('The data of signature {:s} is already exists in collection {:s}.\n Pass.'.format(str(filter_), c_name))
        return results
    return db_handler(db_name)[c_name].insert_one(data)


def insert_many_if_not_exists(db_name, c_name, filter_, data):
    results = db_handler(db_name)[c_name].find_one(filter_)
    if results is not None:
        print('The data of signature {:s} is already exists in collection {:s}.\n Pass.'.format(str(filter_), c_name))
        return results
    return db_handler(db_name)[c_name].insert_many(data)


def replace_one_if_exists(db_name, c_name, filter_, data):
    results = db_handler(db_name)[c_name].find_one_and_replace(filter_, data, upsert=True,
                                                               return_document=ReturnDocument.AFTER)
    if results is not None:
        print('WARN: a document with signature {:s} in the collection {:s} of db {:s} has been replaced'
              .format(str(filter_), c_name, db_name))
    return results


def replace_all_if_exists(db_name, c_name, filter_, data):
    delete_result = delete_many(db_name, c_name, filter_)
    results = db_handler(db_name)[c_name].insert_many(data)
    if delete_result.deleted_count > 0:
        print('''WARN: {:d} documents matching signature {:s} in the collection {:s} of db {:s} has been deleted, \
                 and {:d} documents has been inserted
              '''
              .format(delete_result.deleted_count, str(filter_), c_name, db_name, len(results.inserted_ids)))
    return results


def delete_many(db_name, c_name, filter_):
    print("WARN: Deleting data matching signature {:s} from collection: {:s} of db: {:s}".
          format(str(filter_), c_name, db_name))
    return db_handler(db_name)[c_name].delete_many(filter_)

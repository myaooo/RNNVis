from pymongo import MongoClient
mongo = MongoClient('localhost', 27017)
from rnnvis.db import language_model, sentiment_prediction, db_helper


def seed_db(force=False):
    print('Seeding db: language model')
    language_model.seed_db(force)
    print('Seeding db: sentiment prediction')
    sentiment_prediction.seed_db(force)
    print('Seeding complete')


def get_dataset(name, fields):
    data_info = db_helper.db_hdlr['datasets'].find_one({'name': name})
    if data_info is None:
        print("No dataset with name {:s} exists".format(name))
        return None
    result = db_helper.get_datasets_by_name(name, fields)
    if result is None:
        # result = sentiment_prediction.get_datasets_by_name(name, fields)
        # if result is not None:
        #     print("Data retrieved from sentiment prediction db.")
        # else:
            print("Cannot find any dataset matching name {:s} and fields {:s}".format(name, ", ".join(fields)))
    else:
        print("Data retrieved from db.")
    result.update({'type': data_info['type']})
    return result


class NoDataError(LookupError):
    """
    When data is not find.
    """
    def __init__(self, message=''):
        self.message = message

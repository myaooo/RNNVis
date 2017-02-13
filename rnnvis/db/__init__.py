from pymongo import MongoClient
mongo = MongoClient('localhost', 27017)
from rnnvis.db import language_model, sentiment_prediction


def seed_db():
    print('Seeding db: language model')
    language_model.seed_db()
    print('Seeding db: sentiment prediction')
    sentiment_prediction.seed_db()
    print('Seeding complete')


def get_dataset(name, fields):
    result = language_model.get_datasets_by_name(name, fields)
    if result is None:
        result = sentiment_prediction.get_datasets_by_name(name, fields)
        if result is not None:
            print("Data retrieved from sentiment prediction db.")
        else:
            print("Cannot find any dataset matching name {:s} and fields {:s}".format(name, ", ".join(fields)))
    else:
        print("Data retrieved from language model db.")
    return result

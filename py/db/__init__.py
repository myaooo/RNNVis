from pymongo import MongoClient
mongo = MongoClient('localhost', 27017)
from py.db import language_model, sentiment_prediction


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
    return result

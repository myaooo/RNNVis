from pymongo import MongoClient
mongo = MongoClient('localhost', 27017)
from py.db import language_model

def seed_db():
    print('Seeding db: language model')
    language_model.seed_db()

"""
Use MongoDB to manage all the language modeling datasets
"""

import os
import json
import itertools


import yaml

from rnnvis.utils.io_utils import dict2json, get_path, path_exists
from rnnvis.datasets.data_utils import split
from rnnvis.datasets.text_processor import SSTProcessor, tokenize, tokens2vocab
from rnnvis.datasets.sst_helper import download_sst
from rnnvis.datasets import imdb
from rnnvis.db.db_helper import insert_one_if_not_exists, replace_one_if_exists, \
    store_dataset_by_default, dataset_inserted

# db_name = 'sentiment_prediction'
# # db definition
# collections = {
#     'word_to_id': {'name': (str, 'unique', 'name of the dataset'),
#                    'data': (str, '', 'a json str, encoding word to id mapping')},
#     'id_to_word': {'name': (str, 'unique', 'name of the dataset'),
#                    'data': (list, '', 'a list of str')},
#     'sentences': {'name': (str, '', 'name of the dataset'),
#                   'data': (list, '', 'a list of lists, each list as a sentence of word_ids, data of sst'),
#                   'set': (str, '', 'should be train, valid or test'),
#                   'ids': (list, '', 'the indices in SST')},
#     'eval': {'name': (str, '', 'name of the dataset'),
#              'tag': (str, 'unique', 'a unique tag of hash of the evaluating text sequence'),
#              'data': (list, '', 'a list of word_ids (int), eval text'),
#              'model': (str, '', 'the identifier of the model that the sequence evaluated on'),
#              'records': (list, '', 'a list of ObjectIds of the records')},
#     'record': {'word_id': (int, '', 'word_id in the word_to_id values'),
#                'state': (np.ndarray, 'optional', 'hidden state np.ndarray'),
#                'state_c': (np.ndarray, 'optional', 'hidden state np.ndarray'),
#                'state_h': (np.ndarray, 'optional', 'hidden state np.ndarray'),
#                'output': (np.ndarray, '', 'raw output (unprojected)')},
#     'model': {'name': (str, '', 'identifier of a trained model'),
#               'data_name': (str, '', 'name of the datasets that the model uses')}
# }

# db_hdlr = mongo[db_name]


def store_sst(data_path, name, split_scheme, upsert=False):
    """
    Process and store the ptb datasets to db
    :param data_path:
    :param name:
    :param upsert:
    :return:
    """
    if not path_exists(data_path):
        download_sst(os.path.abspath(os.path.join(data_path, '../')))
    if upsert:
        insertion = replace_one_if_exists
    else:
        insertion = insert_one_if_not_exists
    phrase_path = os.path.join(data_path, "dictionary.txt")
    sentence_path = os.path.join(data_path, "datasetSentences.txt")
    label_path = os.path.join(data_path, "sentiment_labels.txt")
    sentence_split_path = os.path.join(data_path, "datasetSplit.txt")
    processor = SSTProcessor(sentence_path, phrase_path, label_path, sentence_split_path)

    split_data = split(list(zip(processor.ids, processor.labels, range(1, len(processor.labels)+1))),
                       split_scheme.values(), shuffle=True)
    split_data = dict(zip(split_scheme.keys(), split_data))
    sentence_data_ids = processor.split_sentence_ids

    word_to_id_json = dict2json(processor.word_to_id)
    insertion('word_to_id', {'name': name}, {'name': name, 'data': word_to_id_json})
    insertion('id_to_word', {'name': name}, {'name': name, 'data': processor.id_to_word})
    for i, set_name in enumerate(['train', 'valid', 'test']):
        data, ids = zip(*(sentence_data_ids[i]))
        insertion('sentences', {'name': name, 'set': set_name},
                  {'name': name, 'set': set_name, 'data': data, 'ids': ids})

    if 'train' not in split_data:
        print('WARN: there is no train data in the split data!')
    data_dict = {}
    for set_name in ['train', 'valid', 'test']:
        if set_name in split_data:
            data, label, ids = zip(*split_data[set_name])
            # convert label to 1,2,3,4,5
            label = [float(i) for i in label]
            label = [(0 if i <= 0.2 else 1 if i <= 0.4 else 2 if i <= 0.6 else 3 if i <= 0.8 else 4) for i in label]
            data_dict[set_name] = {'data': data, 'label': label, 'ids': ids}
    store_dataset_by_default(name, data_dict)


def store_imdb(data_path, name, n_words=100000, upsert=False):
    if upsert:
        insertion = replace_one_if_exists
    else:
        insertion = insert_one_if_not_exists
    word_to_id, id_to_word = imdb.load_dict(os.path.join(data_path, 'imdb.dict.pkl.gz'), n_words)
    data_label = imdb.load_data(os.path.join(data_path, 'imdb.pkl'), n_words)
    word_to_id_json = dict2json(word_to_id)
    insertion('word_to_id', {'name': name}, {'name': name, 'data': word_to_id_json})
    insertion('id_to_word', {'name': name}, {'name': name, 'data': id_to_word})

    data_dict = {}
    for i, set_name in enumerate(['train', 'valid', 'test']):
        data, label = data_label[i]
        ids = list(range(len(data)))
        # insertion('sentences', {'name': name, 'set': set_name},
        #           {'name': name, 'set': set_name, 'data': data, 'label': label, 'ids': ids})
        data_dict[set_name] = {'data': data, 'label': label, 'ids': ids}
    store_dataset_by_default(name, data_dict)


def store_yelp(data_path, name, n_words=10000, upsert=False):
    if upsert:
        insertion = replace_one_if_exists
    else:
        insertion = insert_one_if_not_exists
    with open(os.path.join(data_path, 'review_label.json'), 'r') as file:
        data = json.load(file)
    training_data, validate_data, test_data = split(data, fractions=[0.8, 0.1, 0.1])
    all_words = []
    reviews = []
    stars = []
    for item in training_data:
        tokenized_review = list(itertools.chain.from_iterable(tokenize(item['review'])))
        reviews.append(tokenized_review)
        stars.append(item['label'])
        all_words.extend(tokenized_review)
    word_to_id, counter, words = tokens2vocab(all_words)

    word_to_id = {k: v+1 for k, v in word_to_id.items() if v < n_words}
    word_to_id['<unk>'] = 0

    id_to_word = [None] * len(word_to_id)
    for word, id_ in word_to_id.items():
        id_to_word[id_] = word

    reviews = [[word_to_id[t] if word_to_id.get(t) else 0 for t in sentence] for sentence in reviews]
    training_data = (reviews, stars)

    tmp_data = []
    for _data in [validate_data, test_data]:
        reviews = []
        stars = []
        for item in _data:
            tokenized_review = list(itertools.chain.from_iterable(tokenize(item['review'])))
            reviews.append([word_to_id[t] if word_to_id.get(t) else 0 for t in tokenized_review])
            stars.append(item['label'])
        tmp_data.append((reviews, stars))
    validate_data = tmp_data[0]
    test_data = tmp_data[1]

    word_to_id_json = dict2json(word_to_id)
    insertion('word_to_id', {'name': name}, {'name': name, 'data': word_to_id_json})
    insertion('id_to_word', {'name': name}, {'name': name, 'data': id_to_word})

    data_names = ['train', 'valid', 'test']
    data_dict = {}
    for i, data_set in enumerate([training_data, validate_data, test_data]):
        data_set = tuple(zip(*sorted(zip(*data_set), key=lambda x: len(x[0]))))
        data, label = data_set
        ids = list(range(len(data)))
        data_dict[data_names[i]] = {'data': data, 'label': label, 'ids': ids}
        insertion('sentences', {'name': name, 'set': data_names[i]},
                  {'name': name, 'set': data_names[i], 'data': data, 'label': label, 'ids': ids})
    store_dataset_by_default(name, data_dict)


def get_dataset_path(name):
    return os.path.join('cached_data', 'sentiment_prediction', name)


def seed_db(force=False):
    """
    Use the `config/datasets/lm.yml` to generate example datasets and store them into db.
    :return: None
    """
    config_dir = get_path('config/db', 'sp.yml')
    with open(config_dir, 'r') as f:
        config = yaml.safe_load(f)['datasets']
    for seed in config:
        print('seeding {:s} data'.format(seed['name']))
        data_dir = get_path('cached_data', seed['dir'])
        seed['scheme'].update({'upsert': force})
        if seed['type'] == 'sst':
            store_sst(data_dir, seed['name'], **seed['scheme'])
        elif seed['type'] == 'imdb':
            store_imdb(data_dir, seed['name'], **seed['scheme'])
        elif seed['type'] == 'yelp':
            store_yelp(data_dir, seed['name'], **seed['scheme'])
        else:
            print('not able to seed datasets with type: {:s}'.format(seed['type']))
            continue
        dataset_inserted(seed['name'], 'sp')


if __name__ == '__main__':
    # store_ptb(get_path('cached_data/simple-examples/data'))
    # store_plain_text(get_path('cached_data/tinyshakespeare.txt'), 'shakespeare',
    # {'train': 0.9, 'valid': 0.05, 'test': 0.05})
    seed_db()
    # data = get_datasets_by_name('sst', ['train', 'valid', 'word_to_id'])
    # train_data = data['train']
    # test_data = data['test']

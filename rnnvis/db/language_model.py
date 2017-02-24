"""
Use MongoDB to manage all the language modeling datasets
"""

import os

import yaml

from rnnvis.utils.io_utils import dict2json, get_path
from rnnvis.datasets.data_utils import load_data_as_ids, split
from rnnvis.datasets.text_processor import PlainTextProcessor
from rnnvis.db.db_helper import replace_one_if_exists, insert_one_if_not_exists, \
    store_dataset_by_default, dataset_inserted

# db_name = 'language_model'
# # db definition
# collections = {
#     'word_to_id': {'name': (str, 'unique', 'name of the dataset'),
#                    'data': (str, '', 'a json str, encoding word to id mapping')},
#     'id_to_word': {'name': (str, 'unique', 'name of the dataset'),
#                    'data': (list, '', 'a list of str')},
#     'train': {'name': (str, 'unique', 'name of the dataset'),
#               'data': (list, '', 'a list of word_ids (int), train data')},
#     'valid': {'name': (str, 'unique', 'name of the dataset'),
#               'data': (list, '', 'a list of word_ids (int), valid data')},
#     'test': {'name': (str, 'unique', 'name of the dataset'),
#              'data': (list, '', 'a list of word_ids (int), test data')},
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
        insertion = replace_one_if_exists
    else:
        insertion = insert_one_if_not_exists

    insertion('word_to_id', {'name': name}, {'name': name, 'data': word_to_id_json})
    insertion('id_to_word', {'name': name}, {'name': name, 'data': id_to_word})
    data_dict = {'train': {'data': train},
                 'valid': {'data': valid},
                 'test': {'data': test}}
    store_dataset_by_default(name, data_dict, upsert)


def store_plain_text(data_path, name, split_scheme, min_freq=1, max_vocab=10000, remove_punct=False, upsert=False):
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
        insertion = replace_one_if_exists
    else:
        insertion = insert_one_if_not_exists
    processor = PlainTextProcessor(data_path, remove_punct=remove_punct)
    processor.tag_rare_word(min_freq, max_vocab)
    split_ids = split(processor.flat_ids, split_scheme.values())
    split_data = dict(zip(split_scheme.keys(), split_ids))
    if 'train' not in split_data:
        print('WARN: there is no train data in the split data!')
    data_dict = {}
    for set_name in ['train', 'valid', 'test']:
        if set_name in split_data:
            data_dict[set_name] = {'data': split_data[set_name]}
    store_dataset_by_default(name, data_dict, upsert)
    insertion('word_to_id', {'name': name}, {'name': name, 'data': dict2json(processor.word_to_id)})
    insertion('id_to_word', {'name': name}, {'name': name, 'data': processor.id_to_word})


def seed_db(force=False):
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
            store_ptb(data_dir, seed['name'], force)
        elif seed['type'] == 'text':
            seed['scheme'].update({'upsert': force})
            store_plain_text(data_dir, seed['name'], **seed['scheme'])
        else:
            print('cannot find corresponding seed functions')
            continue
        dataset_inserted(seed['name'], 'lm')


if __name__ == '__main__':
    seed_db()

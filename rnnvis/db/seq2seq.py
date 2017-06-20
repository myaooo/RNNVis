import os
import yaml
import tarfile

from rnnvis.vendor.data_utils import create_vocabulary, data_to_token_ids, maybe_download, \
    gunzip_file, read_data, initialize_vocabulary, WMT_ENFR_TRAIN_URL, WMT_ENFR_DEV_URL
from rnnvis.utils.io_utils import get_path, file_exists, dict2json
from rnnvis.db.db_helper import insert_one_if_not_exists, replace_one_if_exists, \
    store_dataset_by_default, dataset_inserted, datasets_path

max_train_size = 1000000

_train_path = "giga-fren.release2.fixed"
_dev_path = "newstest2013"


def get_datasets_by_name(name, fields):
    data_path = get_path(datasets_path(name))
    results = {}
    for field in fields:
        if field == 'train':
            path = get_wmt_train_set(data_path)
            results['train'] = read_data(path, max_train_size)
        elif field == 'test' or 'valid':
            path = get_wmt_dev_set(data_path)
            results[field] = read_data(path)


def get_wmt_train_set(directory):
    """Download the WMT en-fr training corpus to directory unless it's there."""
    train_path = os.path.join(directory, _train_path)
    if not file_exists(train_path + ".en"):
        corpus_file = maybe_download(directory, "training-giga-fren.tar",
                                     WMT_ENFR_TRAIN_URL)
        print("Extracting tar file %s" % corpus_file)
        with tarfile.open(corpus_file, "r") as corpus_tar:
            corpus_tar.extractall(directory)
        gunzip_file(train_path + ".en.gz", train_path + ".en")
    return train_path


def get_wmt_dev_set(directory):
    """Download the WMT en-fr training corpus to directory unless it's there."""
    dev_path = os.path.join(directory, _dev_path)
    if not file_exists(dev_path + ".en"):
        dev_file = maybe_download(directory, "dev-v2.tgz", WMT_ENFR_DEV_URL)
        print("Extracting tgz file %s" % dev_file)
        with tarfile.open(dev_file, "r:gz") as dev_tar:
            en_dev_file = dev_tar.getmember("dev/" + _dev_path + ".en")
            en_dev_file.name = _dev_path + ".en"
            dev_tar.extract(en_dev_file, directory)
    return dev_path


def prepare_data(data_dir, train_path, dev_path, vocabulary_size, tokenizer=None):
    """Preapre all necessary files that are required for the training.

      Args:
        data_dir: directory in which the data sets will be stored.
        from_train_path: path to the file that includes "from" training samples.
        to_train_path: path to the file that includes "to" training samples.
        from_dev_path: path to the file that includes "from" dev samples.
        to_dev_path: path to the file that includes "to" dev samples.
        from_vocabulary_size: size of the "from language" vocabulary to create and use.
        to_vocabulary_size: size of the "to language" vocabulary to create and use.
        tokenizer: a function to use to tokenize each data sentence;
          if None, basic_tokenizer will be used.

      Returns:
        A tuple of 6 elements:
          (1) path to the token-ids for "from language" training data-set,
          (2) path to the token-ids for "from language" development data-set,
          (3) path to the "from language" vocabulary file,
      """
    # Create vocabularies of the appropriate sizes.
    vocab_path = os.path.join(data_dir, "vocab%d.from" % vocabulary_size)
    create_vocabulary(vocab_path, train_path, vocabulary_size, tokenizer)

    # Create token ids for the training data.
    train_ids_path = train_path + (".ids%d" % vocabulary_size)
    data_to_token_ids(train_path, train_ids_path, vocab_path, tokenizer)

    # Create token ids for the development data.
    dev_ids_path = dev_path + (".ids%d" % vocabulary_size)
    data_to_token_ids(dev_path, dev_ids_path, vocab_path, tokenizer)

    return train_ids_path, dev_ids_path, vocab_path


def store_wmt_en(data_path, name, vocab_size, upsert=False):
    # train_path = data_paths[0]
    # dev_path = data_paths[1]
    insertion = replace_one_if_exists if upsert else insert_one_if_not_exists

    train_path = get_wmt_train_set(data_path)
    dev_path = get_wmt_dev_set(data_path)
    train_ids_path, dev_ids_path, vocab_path = prepare_data(data_path, train_path, dev_path, vocab_size)
    word_to_id, id_to_word = initialize_vocabulary(vocab_path)
    insertion('word_to_id', {'name': name}, {'name': name, 'data': dict2json(word_to_id)})
    insertion('id_to_word', {'name': name}, {'name': name, 'data': id_to_word})


def seed_db(force=False):
    """
    Use the `config/datasets/lm.yml` to generate example datasets and store them into db.
    :return: None
    """
    config_dir = get_path('config/db', 'seq2seq.yml')
    with open(config_dir, 'r') as f:
        config = yaml.safe_load(f)['datasets']
    for seed in config:
        data_dir = get_path('data', seed['dir'])
        print('seeding {:s} data'.format(seed['name']))
        if seed['type'] == 'wmt':
            store_wmt_en(data_dir, seed['name'], force)
        else:
            print('cannot find corresponding seed functions')
            continue
        dataset_inserted(seed['name'], 'seq2seq', force)

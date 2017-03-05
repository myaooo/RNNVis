"""
Use HDF5 to store evaluations
"""

import hashlib

import h5py

from rnnvis.utils.io_utils import get_path

_root_dir = "_cached/h5"


class H5Table(object):

    def __init__(self, file_name, mode='a'):
        self._file_name = file_name
        self._f = None
        self._mode = mode

    @property
    def file_name(self):
        return self._file_name

    @property
    def f(self):
        if self._f is None:
            self._f = h5py.File(get_path(_root_dir, self.file_name), self.mode)
        return self._f

    @property
    def mode(self):
        return self._mode

    def get_group(self, group_name):
        if group_name not in self.f:
            self.f.create_group(group_name)
        return self.f[group_name]

    def get_dataset(self, dataset_name):
        return self.f[dataset_name]

    def store(self, name, data):
        self.f.create_dataset(name, data=data)

    def __getitem__(self, item):
        return self.f[item]

    def __del__(self):
        self.f.close()


class EvalTable(H5Table):
    def __init__(self, data_name, model_name, mode="a"):
        super(EvalTable, self).__init__(data_name + '-' + model_name + ".h5", mode)
        self.data_name = data_name
        self.model_name = model_name
        self.evals = self.get_group("evals")

    def insert_evaluation(self, eval_text_tokens):
        if isinstance(eval_text_tokens[0], int):
            id_to_word = self['id_to_word']
            eval_ids = eval_text_tokens
            try:
                eval_text_tokens = [id_to_word[i] for i in eval_ids]
            except:
                print('word id of input eval text and dictionary not match!')
                raise
        elif isinstance(eval_text_tokens[0], str):
            word_to_id = self['word_to_id']
            try:
                eval_ids = [word_to_id[token] for token in eval_text_tokens]
            except:
                print('token and dictionary not match!')
                raise
        else:
            raise TypeError('The input eval text should be a list of int or a list of str! But its of type {}'
                            .format(type(eval_text_tokens[0])))


def hash_tag_str(text_list):
    """Use hashlib.md5 to tag a hash str of a list of text"""
    return hashlib.md5(" ".join(text_list).encode()).hexdigest()

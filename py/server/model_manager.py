"""
The backend manager for handling the models
"""

import os

from py.utils.io_utils import get_path, assert_path_exists
from py.procedures import build_model
from py.db import language_model

_config_dir = 'config'
_data_dir = 'cached_data'
_model_dir = 'models'


class ModelManager(object):

    available_models = {
        'PTB': {'config': 'lstm-large3.yml', 'data': 'ptb'},
        'Shakespeare': {'config': 'tinyshakespeare.yml', 'data': 'shakespeare'}
    }

    def __init__(self):
        self.models = {}
        self.data = {}

    def get_model(self, name, train=False):
        if name in self.models:
            return self.models[name]
        else:
            flag = self._load_model(name, train)
            return self.models[name] if flag else None

    def _load_model(self, name, train=False):
        if name in self.available_models:
            config_file = get_path(_config_dir, self.available_models[name]['config'])
            model, _ = build_model(config_file)
            data_name = self.available_models[name]['data']
            data = language_model.get_datasets_by_name(data_name)
            if data is None:
                if name == 'PTB':
                    language_model.store_ptb(get_path('cached_data/simple-examples/data'), data_name)
                elif name == 'Shakespeare':
                    language_model.store_plain_text(get_path('cached_data/tinyshakespeare.txt', data_name),
                                                    'shakespeare', {'train': 0.9, 'valid': 0.05, 'test': 0.05})
                data = language_model.get_datasets_by_name(self.available_models[name]['data'])
            model.add_generator(data['word_to_id'])
            if not train:
                # If not training, the model should already be trained
                assert_path_exists(get_path(_model_dir, model.name))
                model.restore()
            self.models[name] = model
            self.data[name] = data
            return True
        else:
            print('WARN: Cannot find model with name {:s}'.format(name))
            return False

    def model_generate(self, name, seeds, max_branch=1, accum_cond_prob=0.9,
                       min_cond_prob=0.0, min_prob=0.0, max_step=10):
        model = self.models[name]
        return model.generate(seeds, max_branch, accum_cond_prob, min_cond_prob, min_prob, max_step)

if __name__ == '__main__':
    print(os.path.realpath(__file__))

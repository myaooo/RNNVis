"""
The backend manager for handling the models
"""

import os
from py.utils.io_utils import get_path, assert_path_exists
from py.procedures import build_model

_config_dir = 'config'
_data_dir = 'cached_data'
_model_dir = 'models'


class ModelManager(object):

    available_models = {
        'PTB': {'config': 'lstm-large3.yml', 'data': 'simple-examples/data'},
        'Shakespeare': {'config': 'tinyshakespeare.yml', 'data': 'tinyshakespeare.txt'}
    }

    def __init__(self):
        self.models = {}
        self.data_path = {}

    def get_model(self, name, train=False):
        if name in self.available_models:
            config_file = get_path(_config_dir, self.available_models[name]['config'])
            data_path = get_path(_data_dir, self.available_models[name]['data'])
            model = build_model(config_file)
            self.data_path[name] = data_path
            if not train:
                # If not training, the model should already be trained
                assert_path_exists(get_path(_model_dir, model.name))
                model.restore()
            self.models[name] = model
            self.data_path[name] = data_path
        else:
            raise LookupError('Cannot find model with name {:s}'.format(name))
        return model

    # def model_generate(self, name, seed, ):

if __name__ == '__main__':
    print(os.path.realpath(__file__))

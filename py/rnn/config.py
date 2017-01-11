"""
Configurations for RNN models
"""

import yaml


class RNNConfig(object):
    """
    A helper class that specify the configuration of RNN models
    """
    def __init__(self, filename):
        self.filename = filename
        f = open(filename)
        self._dict = yaml.safe_load(f)
        f.close()
        self.__dict__.update(self._dict)

    def save2yaml(self, filename):
        f = open(filename, 'w')
        yaml.dump(self._dict, f)
        f.close()


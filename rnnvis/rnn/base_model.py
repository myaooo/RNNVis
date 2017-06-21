"""
The base interface for Deep RNN Models
"""

class ModelBase(object):

    def train(self, *args, **kwargs):
        raise NotImplementedError("This is the ModelBase class! Use inherited classes!")

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("This is the ModelBase class! Use inherited classes!")

    def save(self, *args, **kwargs):
        raise NotImplementedError("This is the ModelBase class! Use inherited classes!")

    def restore(self, *args, **kwargs):
        raise NotImplementedError("This is the ModelBase class! Use inherited classes!")

    def get_word_from_id(self, ids):
        raise NotImplementedError("This is the ModelBase class! Use inherited classes!")

    def get_id_from_word(self, words):
        raise NotImplementedError("This is the ModelBase class! Use inherited classes!")

class EvaluatorBase(object):
    def evaluate_and_record(self, *args, **kwargs):
        raise NotImplementedError("This is the EvaluatorBase class! Use inherited classes!")
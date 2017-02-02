"""
Helper class and functions for text data using ![NLTK](www.nltk.org)
"""

from collections import Counter
from py.utils.io_utils import path_exists, save2csv, save2text, text2list, csv2list


class PlainTextProcessor(object):
    """
    A helper class that helps to pre-process plain text into tokenized datasets and word_to_ids.
    """

    def __init__(self, text_file, eol=True):
        if not path_exists(text_file):
            raise LookupError("Cannot find file {:s}".format(text_file))
        self.file_path = text_file
        self._tokens = None
        self._flat_tokens = None
        self._ids = None
        self._flat_ids = None
        self._word_to_id = None
        self._id_to_word = None
        self._word_freq = None
        self.rare_list = None
        self._sorted = False
        self.eol = eol

    @property
    def tokens(self):
        if self._tokens is None:
            with open(self.file_path) as f:
                self._tokens = self.tokenize(f.read())
        return self._tokens

    @property
    def flat_tokens(self):
        if self._flat_tokens is None:
            self._flat_tokens = [items for sublist in self.tokens for items in sublist]
        return self._flat_tokens

    @property
    def word_to_id(self):
        if self._word_to_id is None:
            self._word_to_id, self._word_freq, self._id_to_word = self.tokens2vocab(self.flat_tokens, sort=True)
            self._sorted = True
        return self._word_to_id

    @property
    def id_to_word(self):
        if self._id_to_word is None:
            self._word_to_id, self._word_freq, self._id_to_word = self.tokens2vocab(self.flat_tokens, sort=True)
            self._sorted = True
        return self._id_to_word

    @property
    def word_freq(self):
        if self._word_freq is None:
            self._word_to_id, self._word_freq, self._id_to_word = self.tokens2vocab(self.flat_tokens, sort=True)
            self._sorted = True
        return self._word_freq

    @property
    def ids(self):
        if self._ids is None:
            self._ids = [[self.word_to_id[word] for word in sublist] for sublist in self.tokens]
        return self._ids

    @property
    def flat_ids(self):
        if self._flat_ids is None:
            self._flat_ids = [items for sublist in self.ids for items in sublist]
        return self._flat_ids

    def tag_rare_word(self, min_freq=3, max_vocab=10000):
        """
        Annotate rare words with an <unk> tag
        :param min_freq: the minimum frequency that a word is not considered as a rare one
        :param max_vocab: the maximum vocabulary size, including <unk> tag
        :return: None
        """

        if len(self.id_to_word) > max_vocab:
            # must have unk tag
            max_vocab -= 1
            rare_num = len(self.id_to_word) - max_vocab
        else:
            max_vocab = len(self.id_to_word)
            rare_num = 0

        for i in reversed(range(max_vocab)):
            # iterate in the reversed order of their frequency
            word = self.id_to_word[i]
            if self.word_freq[word] < min_freq:
                rare_num += 1
            else:
                break
        if rare_num == 0:  # no need to tag rare words
            return

        rare_list = []
        unk_freq = 0
        for i in range(len(self.id_to_word)-rare_num, len(self.id_to_word)):
            word = self.id_to_word[i]
            rare_list.append(word)
            unk_freq += self.word_freq.pop(word)
        self.word_freq['<unk>'] = unk_freq
        # need to re-sort
        count_pairs = sorted(self.word_freq.items(), key=lambda x: (-x[1], x[0]))
        self._id_to_word, _ = list(zip(*count_pairs))
        self._word_to_id = dict(zip(self.id_to_word, range(len(self.id_to_word))))
        # update tokens and ids
        rare_set = set(rare_list)
        for sublist in self.tokens:
            for i, token in enumerate(sublist):
                if token in rare_set:
                    sublist[i] = '<unk>'
        # These will be reconstructed when accessed
        self._ids = None
        self._flat_ids = None
        self._flat_tokens = None

    def save(self, name=None):
        """
        This save function will save 2 files, a xxx.dict.csv file as word_to_id lookup table,
         and a xxx.ids file as converted ids for fast loading and training
        :param name: an optional directory other than the original file_path
        :return:
        """
        if name is None:
            name = self.file_path
        ids = name+'.ids.csv'
        print('saving ids to {:s}'.format(ids))
        save2csv(self.ids, ids, delimiter=' ')
        dictionary = name+'.dict.csv'
        print('saving dictionary to {:s}'.format(dictionary))
        save2csv([[word, i] for word, i in self.word_to_id.items()], dictionary, " ")

    @staticmethod
    def load(text_file):
        ids = text_file + '.ids.csv'
        dictionary = text_file + '.dict.csv'
        proc = PlainTextProcessor(text_file)
        proc._ids = save2csv(ids, " ")
        dict_list = csv2list(dictionary, " ")
        proc._word_to_id = {e[0]: e[1] for e in dict_list}
        return proc

    @staticmethod
    def tokenize(str_stream, eol=True):
        # do lazy import coz import nltk is very slow
        import nltk
        try:
            nltk.data.load('tokenizers/punkt/english.pickle')
        except LookupError:
            print('punct resource not found, using nltk.download("punkt") to download resource data...')
            nltk.download('punkt')
        tokens = [nltk.word_tokenize(t) for t in nltk.sent_tokenize(str_stream.lower())]
        if eol:
            for token in tokens:
                token[-1] = '<eos>'
        return tokens

    @staticmethod
    def tokens2vocab(tokens, sort=True):
        counter = Counter(tokens)
        if sort:
            count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
            words, _ = list(zip(*count_pairs))
        else:
            words, _ = list(zip(*counter.items()))
        word_to_id = dict(zip(words, range(len(words))))
        return word_to_id, counter, words


if __name__ == "__main__":

    processor = PlainTextProcessor('../../cached_data/tinyshakespeare.txt')
    processor.tag_rare_word(2, 10000)
    processor.save()

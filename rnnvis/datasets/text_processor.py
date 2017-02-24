"""
Helper class and functions for text data using ![NLTK](www.nltk.org)
"""

import functools
from collections import Counter
from rnnvis.utils.io_utils import path_exists, lists2csv, save2text, text2list, csv2list, get_path


def lazy_property(func):
    attribute = '_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if (not hasattr(self, attribute)) or getattr(self, attribute) is None:
            setattr(self, attribute, func(self))
        return getattr(self, attribute)

    return wrapper


class PlainTextProcessor(object):
    """
    A helper class that helps to pre-process plain text into tokenized datasets and word_to_ids.
    tokens: the tokens attr is a list of lists, each nested list contains word tokens of a sentence,
        the tokens are extracted using nltk
    flat_tokens:
    """

    def __init__(self, text_file, eos=True, remove_punct=False):
        """
        a plainTextProcessor converts english texts into serialized word tokens, creates dictionary and so on.
        :param text_file: the path of the .txt file
        :param eos: end of sentence, whether convert dot into <eos> tag
        :param remove_punct, whether remove other punctuations like ",", ":", etc.
        """
        if not path_exists(text_file):
            raise LookupError("Cannot find file {:s}".format(text_file))
        self.file_path = text_file
        self.eos = eos
        self.remove_punct = remove_punct
        self.rare_list = None
        self._word_to_id = None
        self._id_to_word = None
        self._word_freq = None
        self._sorted = False

    @lazy_property
    def tokens(self):
        with open(self.file_path) as f:
            return tokenize(f.read(), self.eos, self.remove_punct)

    @lazy_property
    def flat_tokens(self):
        return [items for sublist in self.tokens for items in sublist]

    @property
    def word_to_id(self):
        if self._word_to_id is None:
            self._word_to_id, self._word_freq, self._id_to_word = tokens2vocab(self.flat_tokens, sort=True)
            self._sorted = True
        return self._word_to_id

    @property
    def id_to_word(self):
        if self._id_to_word is None:
            self._word_to_id, self._word_freq, self._id_to_word = tokens2vocab(self.flat_tokens, sort=True)
            self._sorted = True
        return self._id_to_word

    @property
    def word_freq(self):
        if self._word_freq is None:
            self._word_to_id, self._word_freq, self._id_to_word = tokens2vocab(self.flat_tokens, sort=True)
            self._sorted = True
        return self._word_freq

    @lazy_property
    def ids(self):
        return [[self.word_to_id[word] for word in sublist] for sublist in self.tokens]

    @lazy_property
    def flat_ids(self):
        return [items for sublist in self.ids for items in sublist]

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
        lists2csv(self.ids, ids, delimiter=' ')
        dictionary = name+'.dict.csv'
        print('saving dictionary to {:s}'.format(dictionary))
        lists2csv([[word, i] for word, i in self.word_to_id.items()], dictionary, " ")

    @staticmethod
    def load(text_file):
        ids = text_file + '.ids.csv'
        dictionary = text_file + '.dict.csv'
        proc = PlainTextProcessor(text_file)
        proc._ids = lists2csv(ids, " ")
        dict_list = csv2list(dictionary, " ")
        proc._word_to_id = {e[0]: e[1] for e in dict_list}
        return proc


def isfloat(s):
    try:
        float(s)
        if s == 'nan':
            return False
        return True
    except ValueError:
        return False


__punct_set = {':', ';', '--', ',', "'"}


def tokenize(str_stream, eos=True, remove_punct=False):
    """
    Given a str or str_stream (f.read()) convert the str to a list of sentences,
        e.g.: [[word, word], [word, word, ...], ...]
    :param str_stream: a str or a str_stream
    :param eos: wether turns '.' into <eos> tag
    :param remove_punct: wether to remove punctuations: ':', ';', '--', ',', "'"
    :return: a list of sentences, each sentence is a list of words (str)
    """
    # do lazy import coz import nltk is very slow
    import nltk
    try:
        nltk.data.load('tokenizers/punkt/english.pickle')
    except LookupError:
        print('punct resource not found, using nltk.download("punkt") to download resource data...')
        nltk.download('punkt')
    tokens = [nltk.word_tokenize(t) for t in nltk.sent_tokenize(str_stream.lower())]
    # tag number
    tokens = [['N' if isfloat(t) else t for t in sublist] for sublist in tokens]
    if eos:
        for token in tokens:
            token[-1] = '<eos>'
    if remove_punct:
        tokens = [[t for t in sublist if t not in __punct_set] for sublist in tokens]
    return tokens


def tokens2vocab(tokens, sort=True):
    """
    Given a list of tokens (words), get the word_to_id dict, as well as word_freq, and id_to_word
    :param tokens: a list of words of type str
    :param sort: whether to sort the words according to their frequency
    :return: a tuple (word_to_id, word_freq, id_to_word)
    """
    counter = Counter(tokens)
    print("vocab size: {:d}".format(len(counter)))
    if sort:
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
    else:
        words, _ = list(zip(*counter.items()))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id, counter, words


class SSTProcessor(PlainTextProcessor):

    def __init__(self, sentence_file, diction_file, sst_labels, sentence_split_file):
        """"""
        super(SSTProcessor, self).__init__(diction_file)
        self.label_path = sst_labels
        self.sentence_path = sentence_file
        self.sentence_split_path = sentence_split_file

    @lazy_property
    def tokens(self):
        phrase_id_list = csv2list(self.file_path, delimiter='|', skip=1, encoding='utf-8')
        phrase_id_list = [(e[0], int(e[1])) for e in phrase_id_list]
        phrases = [None] * len(phrase_id_list)
        # print("phrase num: {:d}".format(len(phrase_id_list)))
        # print("max_id: {:d}".format(max([e[1] for e in phrase_id_list])))
        for phrase, id_ in phrase_id_list:
            phrases[id_-1] = phrase
        return [phrase.split(sep=' ') for phrase in phrases]

    @lazy_property
    def sentence_tokens(self):
        id_sentence_lists = csv2list(self.sentence_path, delimiter='\t', skip=1, encoding='utf-8')
        sentences = [e[1] for e in id_sentence_lists]
        return [sen.split(sep=' ') for sen in sentences]

    @lazy_property
    def sentence_ids(self):
        return [[self.word_to_id[word] for word in sublist] for sublist in self.sentence_tokens]

    @lazy_property
    def split_sentence_ids(self):
        sentence_label_lists = csv2list(self.sentence_split_path, delimiter=',', skip=1)
        sentence_label_lists = [(int(e[0]), int(e[1])) for e in sentence_label_lists]
        train = []
        valid = []
        test = []
        for sentence_id, split_label in sentence_label_lists:
            if split_label == 1:
                target = train
            elif split_label == 2:
                target = valid
            elif split_label == 3:
                target = test
            else:
                raise ValueError('sentence split labels should be 1,2,3, but get {:d}'.format(split_label))
            target.append((self.sentence_ids[sentence_id - 1], sentence_id))
        return train, valid, test

    @lazy_property
    def labels(self):
        id_label_list = csv2list(self.label_path, delimiter='|', skip=1)
        return [e[1] for e in id_label_list]

    def save(self, name=None):
        """
        This save function will save 2 files, a xxx.dict.csv file as word_to_id lookup table,
         and a xxx.ids file as converted ids for fast loading and training
        :param name: an optional directory other than the original file_path
        :return:
        """
        if name is None:
            name = self.file_path
        ids = name + '.ids.csv'
        print('saving ids to {:s}'.format(ids))
        lists2csv(self.ids, ids, delimiter=' ')
        dictionary = name + '.dict.csv'
        print('saving dictionary to {:s}'.format(dictionary))
        lists2csv([[word, i] for word, i in self.word_to_id.items()], dictionary, " ", encoding='utf-8')


if __name__ == "__main__":

    # processor = PlainTextProcessor('../../cached_data/tinyshakespeare.txt')
    # processor.tag_rare_word(2, 10000)
    # processor.save()
    processor = SSTProcessor(get_path('./cached_data/stanfordSentimentTreebank', 'datasetSentences.txt'),
                             get_path('./cached_data/stanfordSentimentTreebank', 'dictionary.txt'),
                             get_path('./cached_data/stanfordSentimentTreebank', 'sentiment_lables.txt'))
    # tokens = processor.tokens
    # processor.ids = None
    # word_to_id = processor.word_to_id
    # sentences = processor.sentence_tokens
    processor.save()

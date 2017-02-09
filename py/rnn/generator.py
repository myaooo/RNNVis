"""
A generator take a seed word and generate sequence / sequence tree using the trained model
"""

import numpy as np
from . import rnn
from . import losses
from py.utils.io_utils import dict2json
from py.utils.tree import TreeNode, Tree


class GenerateNode(TreeNode):
    """
    Node structure to store generation tree of a RNN
    """
    def __init__(self, word_id, prob, cond_prob):
        super().__init__()
        self.word_id = word_id
        self.word = None
        self.prob = prob
        self.cond_prob = cond_prob


class Generator(object):

    def __init__(self, rnn_):
        """
        :param rnn_: An RNN instance
        """
        assert isinstance(rnn_, rnn.RNN)
        self._rnn = rnn_
        self.model = rnn_.unroll(1, 1, name='generator')

    def generate(self, sess, seeds, logdir=None, max_branch=3, accum_cond_prob=0.9,
                 min_cond_prob=0.1, min_prob=0.001, max_step=10, neg_word_ids=None):
        """
        Generate sequence tree with given seed (a word_id) and certain requirements
        Note that the method always try to generate as much branches as possible.
        :param sess: the sess to run the model
        :param seeds: a list of word_id or a list of words
        :param logdir: the file path to save the generating tree
        :param max_branch: the maximum number of branches at each node
        :param accum_cond_prob: the maximum accumulate conditional probability of the following branches
        :param min_cond_prob: the minimum conditional probability of each branch
        :param min_prob: the minimum probability of a branch (note that this indicates a multiplication along the tree)
        :param max_step: the step to generate
        :param neg_word_ids: a set of neglected words' ids.
        :return: if logdir is None, returns a dict object representing the tree. if logdir is not None, return None
        """

        model = self.model
        model.reset_state()
        # Initialize the tree and inserts the seeds node
        tree = Tree()
        # converts words into ids
        if (not isinstance(seeds, list)) or len(seeds) < 1:
            raise ValueError("seeds should be a list of words or ids")
        if isinstance(seeds[0], str):
            _seeds = self.get_id_from_word(seeds)
            seeds = _seeds
        parent = GenerateNode(seeds[0], 1.0, 1.0)
        tree.add_node(parent, None)
        for seed in seeds[1:]:
            node = GenerateNode(seed, 1.0, 1.0)
            tree.add_node(node, parent)
            parent = node
        neg_word_ids = set(neg_word_ids)  # converts to set for easier `in` statement

        def _generate(node, step):
            if step > max_step:  # Already at the maximum generating step
                return
            prev_prob = node.prob
            # The second inputs is just to hold place. See the implementation of model.run()
            evals, _ = model.run(np.array(node.word_id).reshape(1, 1), None, 1, sess,
                                 eval_ops={'projected': model.projected_outputs})
            outputs = evals['projected'][0].reshape(-1)
            # do softmax so that outputs represents probs
            outputs = losses.softmax(outputs)
            # Get sorted k max probs and their ids,
            # since we will neglect some of them latter, we first get a bit more of the top k
            max_id = np.argpartition(-outputs, max_branch)[:(max_branch+len(neg_word_ids))]
            del_ids = []
            for i, id_ in enumerate(max_id):
                if int(id_) in neg_word_ids:
                    del_ids.append(i)
            max_id = np.delete(max_id, del_ids)
            max_id = max_id[:max_branch]
            cond_probs = outputs[max_id]
            # Sort the cond_probs for later filtering use
            sort_indice = np.argsort(-cond_probs)
            max_id = max_id[sort_indice]
            cond_probs = cond_probs[sort_indice]
            prob_sum = np.sum(cond_probs)
            # do filtering according to accum_prob
            while len(cond_probs) > 0:
                if accum_cond_prob > prob_sum:
                    break
                prob_sum -= cond_probs[-1]
                cond_probs = cond_probs[:-1]
            # do filtering according to min_cond_prob
            while len(cond_probs) > 0:
                if cond_probs[-1] > min_cond_prob:
                    break
                cond_probs = cond_probs[:-1]
            while len(cond_probs) > 0:
                if cond_probs[-1] * prev_prob > min_prob:
                    break
                # the probability of this branch is too small
                cond_probs = cond_probs[:-1]
            if len(cond_probs) == 0:  # No available nodes to generate
                return
            max_id = max_id[:len(cond_probs)]
            for word_id, cond_prob in zip(max_id, cond_probs):
                new_node = GenerateNode(int(word_id), float(cond_prob*prev_prob), float(cond_prob))
                tree.add_node(new_node, node)
            for child in tree.get_children(node):
                _generate(child, step+1)

        _generate(parent, len(seeds))
        for node in tree.nodes():
            node.word = self.get_word_from_id(node.word_id)
        # print(tree.as_dict())
        return dict2json(tree.as_dict(), logdir)

    def get_word_from_id(self, ids):
        """
        Retrieve the words by ids
        :param ids: a numpy.ndarray or a list or a python int
        :return: a list of words
        """
        return self._rnn.get_word_from_id(ids)

    def get_id_from_word(self, words):
        """
        Retrieve the ids from words
        :param words: a list of words
        :return: a list of corresponding ids
        """
        return self._rnn.get_id_from_word(words)

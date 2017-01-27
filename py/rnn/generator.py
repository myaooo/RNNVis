"""
A generator take a seed word and generate sequence / sequence tree using the trained model
"""

import numpy as np
from . import rnn
from py.utils.io_utils import save2json
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

    def __init__(self, rnn_, word_to_id):
        """
        :param rnn_: An RNN instance
        """
        assert isinstance(rnn_, rnn.RNN)
        self.model = rnn_.unroll(1, 1, name='generator')
        self.word_to_id = word_to_id
        self.id_to_word = {id_: word for word, id_ in word_to_id.items()}

    def get_word_from_id(self, ids):
        """
        Retrieve the words by ids
        :param ids: a numpy.ndarray or a list or a python int
        :return: a list of words
        """
        if isinstance(ids, int):
            return self.id_to_word[ids]
        words = []
        for i in ids:
            words.append(self.id_to_word[i])
        return words

    def generate(self, sess, seed, logdir, max_branch=3, accum_cond_prob=0.9,
                 min_cond_prob=0.1, min_prob=0.001, max_step=10):
        """
        Generate sequence tree with given seed (a word_id) and certain requirements
        Note that the method always try to generate as much branches as possible.
        :param sess: the sess to run the model
        :param seed: a word_id
        :param logdir: the file path to save the generating tree
        :param max_branch: the maximum number of branches at each node
        :param accum_cond_prob: the maximum accumulate conditional probability of the following branches
        :param min_cond_prob: the minimum conditional probability of each branch
        :param min_prob: the minimum probability of a branch (note that this indicates a multiplication along the tree)
        :param max_step: the step to generate
        :return:
        """

        model = self.model
        model.init_state(sess)
        tree = Tree()
        tree.add_node(GenerateNode(seed, 1.0, 1.0), None)

        def _generate(node, step):
            if step > max_step:  # Already at the maximum generating step
                return
            prev_prob = node.prob
            # The second inputs is just to hold place. See the implementation of model.run()
            evals, _ = model.run(np.array(node.word_id).reshape(1, 1), None, 1, sess,
                                 eval_ops={'projected': model.projected_outputs})
            outputs = evals['projected'][0].reshape(-1)
            outputs = rnn.softmax(outputs)
            # Get sorted k max probs and their ids
            max_id = np.argpartition(-outputs, max_branch)[:max_branch]
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
                new_node = GenerateNode(word_id, cond_prob*prev_prob, cond_prob)
                tree.add_node(new_node, node)
            for child in node.children:
                _generate(child, step+1)

        _generate(tree.get_root(), 0)
        for node in tree.nodes():
            node.word = self.get_word_from_id(node.word_id)
        save2json(tree.as_dict(), logdir)


"""
A generator take a seed word and generate sequence / sequence tree using the trained model
"""

import time

import numpy as np
import tensorflow as tf

from . import rnn
from . import losses
from rnnvis.utils.io_utils import dict2json
from rnnvis.utils.tree import TreeNode, Tree


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
        self.model = rnn_.unroll(20, 1, name='generator{:d}'.format(len(rnn_.models)))

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
        :param neg_word_ids: a set of neglected words or words' ids.
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
        if neg_word_ids is None:
            neg_word_ids = {}
        elif isinstance(neg_word_ids[0], str):
            neg_word_ids = self.get_id_from_word(neg_word_ids)
        elif not isinstance(neg_word_ids[0], int):
            raise TypeError("neg_word_ids should be a iterable object containing words as word tokens or word ids!")
        neg_word_ids = set(neg_word_ids)  # converts to set for easier `in` statement
        # print(neg_word_ids)

        buffer_size = self.model.batch_size

        def _generate(_buffer, step):
            if step > max_step:  # Already at the maximum generating step
                return
            if len(_buffer) > buffer_size:
                _b = []
                for j in range(0, len(_buffer), buffer_size):
                    _b += _generate(_buffer[j:(j+buffer_size)], step)
                return _b
            nodes, states = zip(*_buffer)
            word_ids = [n.word_id for n in nodes]
            prev_probs = [n.prob for n in nodes]
            states = _pack_list_to_states(states, buffer_size)
            # padding -1s to make sure that shape match
            word_ids += [-1] * (buffer_size - len(word_ids))

            # prev_prob = node.prob
            # The second inputs is just to hold place. See the implementation of model.run()
            model.current_state = states
            evals, _ = model.run(np.array(word_ids).reshape(buffer_size, 1), None, 1, sess,
                                 eval_ops={'projected': model.projected_outputs})
            new_buffer = []
            # shape: [batch_size * num_steps, project_size]
            batch_outputs = evals['projected'][0]
            current_states = _convert_state_to_list(model.current_state, len(_buffer))

            def _filter_and_append(outputs, pos):

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
                    if cond_probs[-1] * prev_probs[pos] > min_prob:
                        break
                    # the probability of this branch is too small
                    cond_probs = cond_probs[:-1]
                if len(cond_probs) == 0:  # No available nodes to generate
                    return
                max_id = max_id[:len(cond_probs)]
                for word_id, cond_prob in zip(max_id, cond_probs):
                    new_node = GenerateNode(int(word_id), float(cond_prob*prev_probs[pos]), float(cond_prob))
                    tree.add_node(new_node, nodes[pos])
                for child in tree.get_children(nodes[pos]):
                    new_buffer.append((child, current_states[pos]))

            for j in range(len(_buffer)):
                _filter_and_append(batch_outputs[j], j)

            return new_buffer

        start_time = time.time()
        model.init_state(sess)
        buffer = [(parent, _convert_state_to_list(model.current_state, 1)[0])]
        for i in range(len(seeds), max_step):
            buffer = _generate(buffer, i)
            if len(buffer) == 0:
                break
        print("total_time: {:f}s, speed: {:f}wps".format(time.time() - start_time, len(tree)/(time.time()-start_time)))
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


def _convert_state_to_list(state, size):
    states = []
    for i in range(size):
        st = []
        for s in state:  # multilayers
            # s is tuple
            if isinstance(s, tf.nn.rnn_cell.LSTMStateTuple):
                st.append((s.c[i], s.h[i]))
            else:
                st.append(s)
        states.append(st)
    return states


def _pack_list_to_states(state_list, batch_size):
    states = []
    for i in range(len(state_list[0])):
        if len(state_list[0][i]) == 2:
            cs = []
            hs = []
            for j in range(len(state_list)):
                cs.append(state_list[j][i][0])
                hs.append(state_list[j][i][1])
            for j in range(len(state_list), batch_size):
                cs.append(np.zeros(state_list[0][i][0].shape, dtype=np.float32))
                hs.append(np.zeros(state_list[0][i][1].shape, dtype=np.float32))
            state = tf.nn.rnn_cell.LSTMStateTuple(np.vstack(cs), np.vstack(hs))
        else:
            s = []
            for j in range(len(state_list)):
                s.append(state_list[j][i])
            for j in range(len(state_list), batch_size):
                s.append(np.zeros(state_list[0][i].shape, dtype=np.float32))
            state = np.vstack(s)
        states.append(state)
    return states
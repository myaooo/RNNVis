import uuid
import queue

from py.utils.tree import TreeNode, Tree

class XTree(object):
    """"This class is just a wrap of class SuffixTree which provide both prefix and suffix
    """
    def __init__(self, data, sentence_delimiter='\n'):
        self._data = data
        self._prefix_data, self._suffix_data = self.data_preprocessing(
            self._data, sentence_delimiter)

        self._prefix_tree = SuffixTree(self._prefix_data)
        self._suffix_tree = SuffixTree(self._suffix_data)

    def fetch_prefix_suffix(self, gram):
        if isinstance(gram, str):
            gram = gram.split()
        prefix_result = self._prefix_tree.search_suffix(list(reversed(gram)))
        suffix_result = self._suffix_tree.search_suffix(gram)

        q = queue.Queue()
        q.put(prefix_result)
        while not q.empty():
            node = q.get()
            prefix_position = (node['position'][0],
                               len(self._prefix_data[node['position'][0]]) -
                               node['position'][1] - len(node['data']))
            node['position'] = prefix_position
            node['data'] = ' '.join(list(reversed(node['data'])))
            for child in node['children']:
                q.put(child)

        q.put(suffix_result)
        while not q.empty():
            node = q.get()
            node['data'] = ' '.join(node['data'])
            for child in node['children']:
                q.put(child)

        return prefix_result, suffix_result

    def data_preprocessing(self, data, sentence_delimiter='\n'):
        """ Preprocess the raw data to produce prefix and suffix data
        """
        lines = data.split(sentence_delimiter)
        prefix_data = []
        for i, line in enumerate(lines):
            words = line.split()
            words = list(reversed(words))
            words.append('<bos_' + str(i) + '>')
            prefix_data.append(words)
        suffix_data = []
        for i, line in enumerate(lines):
            words = line.split()
            words.append('<eos_' + str(i) + '>')
            suffix_data.append(words)

        return prefix_data, suffix_data


class SuffixTree(Tree):
    """This class is used to construct suffix tree
    """

    def __init__(self, data):
        Tree.__init__(self)
        root_node = SuffixTreeNode(data=[], position=(-1, -1))
        self.add_node(root_node)
        self._root = self.get_root()
        self._data = data

        self._leaves = set()
        self._status = None
        self._remainder = 0
        self._previous_node_added = None
        self._reverse_suffix_links = {}
        self._tmp_str = None

        self._words = None
        self._sentence_num = -1

        self.construct(self._data)
        self._tree_dict = self.to_dict(self.get_root())

    def construct(self, data):
        for i, line in enumerate(data):
            self._status = {'active_node': self._root,
                            'active_edge': None,
                            'active_length': 0
                            }
            self._leaves = set()
            self._remainder = 0
            self._words = line
            self._sentence_num = i

            for j, word in enumerate(line):
                for leaf in self._leaves:
                    self.get_node(leaf).append_suffix(word)
                self._remainder += 1
                self.add_to_tree(j)
                self._previous_node_added = None
                # self._tmp_str = self.tree_str(self.get_root())
            # print(self.to_dict(self.get_root()))

    def search_suffix(self, gram):
        if isinstance(gram, str):
            gram = gram.split()
        gram_node, position = self.find_node(self.get_root(), gram)
        if position < 0:
            return None
        tree_dict = self.to_dict(gram_node)
        tree_dict['data'] = ' '.join(gram_node.data[position:])
        return tree_dict

    def to_dict(self, node):
        node_info = {}
        node_info['data'] = node.data if not node.is_root() else ['#root#']
        node_info['position'] = node.position
        node_info['children'] = []
        for child_id in node.children_id:
            child_node = self.get_node(child_id)
            node_info['children'].append(self.to_dict(child_node))
        return node_info


    def find_node(self, start_node, gram):
        if not start_node.children_prefix(gram[0]):
            return start_node, -1

        edge = self.get_node(start_node.children_prefix(gram[0]))
        for i in range(min(len(gram), len(edge.data))):
            if gram[i] != edge.data[i]:
                return start_node, -1
        i += 1
        if i == len(gram):
            return edge, i
        elif i == len(edge.data):
            return self.find_node(edge, gram[i:])

    def tree_str(self, node):
        if len(node.children_id) < 1:
            return ' '.join(node.data)
        strs = []
        for child_id in node.children_id:
            strs.append(self.tree_str(self.get_node(child_id)))
        key_ = ' '.join(node.data) if not node.is_root() else 'root'
        return {key_: strs}

    def add_node(self, node, parent=None):
        if not isinstance(node, TreeNode):
            raise TypeError("arg node should be of type TreeNode, but it's of type {:s}".format(type(node)))
        if node.id in self._dict:
            raise ValueError("node with id {:s} already exists!".format(node.id))
        if parent is None:
            if self.root_id is not None:
                raise ValueError("Attempt to add a multiple root! The tree already has a root.")
            self.root_id = node.id
        else:
            parent = self.get_node(parent)
            if parent is None:
                raise ValueError("the given parent is not in the tree!")
            parent.children_id.add(node.id)
            parent.add_child(node)
            node.parent_id = parent.id
        self._dict[node.id] = node

    def add_to_tree(self, i):
        if self._remainder == 0:
            return

        # word = words[i]
        # if self._status['active_node'].is_root():
        #     tmp_word = words[i - self._remainder + 1]
        # else:
        #     tmp_word = words[i - self._status['active_length']]
        # word = self._words[i - self._remainder + 1] if self._status['active_node'].is_root() else \
        #     self._words[i - self._status['active_length']]
        word_pos = i - self._remainder + 1 if self._status['active_node'].is_root() else \
            i - self._status['active_length']
        word = self._words[word_pos]
        if self._status['active_edge'] is None:
            if not self._status['active_node'].children_prefix(word):
                node = SuffixTreeNode(data=[word], position=(self._sentence_num, word_pos))
                self.add_node(node, parent=self._status['active_node'])
                self._leaves.add(node.id)
                self._remainder -= 1
                if not self._status['active_node'].is_root():
                    self._status['active_node'] = self.get_node(
                        self._status['active_node'].suffix_link) if \
                        self._status['active_node'].suffix_link else self.get_root()
                    # if self._previous_node_added is not None:
                    #     self._previous_node_added.suffix_link = node.id
                    # self._previous_node_added = node
                    self._previous_node_added = None
                    self.add_to_tree(i)

            else:
                edge_node_id = self._status['active_node'].children_prefix(word)
                edge_node = self.get_node(edge_node_id)
                self._status['active_edge'] = edge_node
                self._status['active_length'] = 0
                counter = 0
                while counter+1 <= self._remainder and self._status['active_edge'] and self._status['active_edge'].data[
                    self._status['active_length']] == self._words[i - self._remainder + counter + 1]:

                    self._status['active_length'] += 1
                    counter += 1
                    if len(self._status['active_edge'].data) == self._status['active_length']:
                        self._status['active_node'] = self._status['active_edge']
                        self._status['active_edge'] = self._status['active_node'].children_prefix(
                            self._words[i - self._remainder + counter + 2]) if counter+2 <= self._remainder else None
                        self._status['active_length'] = 0

                if counter == self._remainder:
                    # contain all infomation
                    return
                self.add_to_tree(i)

        elif self._status['active_edge'].data[self._status['active_length']] == self._words[i]:
            self._status['active_length'] += 1
            if len(self._status['active_edge'].data) == self._status['active_length']:
                self._status['active_node'] = self._status['active_edge']
                self._status['active_edge'] = None
                self._status['active_length'] = 0

        elif self._status['active_edge'].data[self._status['active_length']] != self._words[i]:
            new_node = SuffixTreeNode(data=[self._words[i]], position=(self._sentence_num, i))
            other_node = self.split_node(self._status['active_edge'], self._status['active_length'])
            self.add_node(new_node, self._status['active_edge'])
            self.add_node(other_node, self._status['active_edge'])
            self._leaves.add(new_node.id)

            if self._previous_node_added is not None:
                self._previous_node_added.suffix_link = self._status['active_edge'].id
            self._previous_node_added = self._status['active_edge']

            self._remainder -= 1

            # update active node and active edge
            if self._status['active_node'].is_root():
                self._status['active_length'] -= 1
                edge_node_id = self._status['active_node'].children_prefix(self._words[i - self._status['active_length']])
                self._status['active_edge'] = self.get_node(edge_node_id) if edge_node_id else None
                while self._status['active_edge'] and self._status['active_length'] >= len(self._status['active_edge'].data):
                    self._status['active_node'] = self._status['active_edge']
                    self._status['active_length'] -= len(self._status['active_edge'].data)
                    edge_node_id = self._status['active_node'].children_prefix(self._words[i - self._status['active_length']])
                    self._status['active_edge'] = self.get_node(edge_node_id) if edge_node_id else None

                self.add_to_tree(i)
            elif self._status['active_node'].suffix_link:
                self._status['active_node'] = self.get_node(
                    self._status['active_node'].suffix_link)
                edge_node_id = self._status['active_node'].children_prefix(self._words[i - self._status['active_length']])
                self._status['active_edge'] = self.get_node(edge_node_id)
                self.add_to_tree(i)
            else:
                self._status['active_node'] = self.get_root()
                self._status['active_edge'] = None
                self.add_to_tree(i)

    def split_node(self, node, position_):
        new_node = SuffixTreeNode(data=node.data[position_:],
                                  position=(node.position[0], node.position[1]+position_))
        node.data = node.data[:position_]
        for child_id in node.children_id:
            child_node = self.get_node(child_id)
            child_node.parent_id = new_node.id
        new_node.children_id = node.children_id
        new_node.child_prefix_dict = node.child_prefix_dict
        node.delete_all_children()
        if node.id in self._leaves:
            self._leaves.remove(node.id)
            self._leaves.add(new_node.id)
        return new_node

class SuffixTreeNode(TreeNode):
    '''Suffix tree node class, inherit from TreeNode
    '''

    def __init__(self, data, position, id_=None, parent_id=None):
        if id_ is None:
            id_ = uuid.uuid1().hex
        TreeNode.__init__(self, id_=id_, parent_id=parent_id)
        self._data = data
        self._child_prefix_dict = {}
        self._suffix_link = None
        self._position = position

    def first_element(self):
        """Return the first element of node data
        """
        if not isinstance(self._data, list):
            raise ValueError('The node data should be a list of words')
        if len(self._data) < 1:
            raise ValueError('The length of node data should at least be 1')
        return self._data[0]

    def children_prefix(self, word):
        return self._child_prefix_dict.get(word)

    def append_suffix(self, word):
        self._data.append(word)

    def add_child(self, child_node):
        self._child_prefix_dict[child_node.first_element()] = child_node.id
        self._children_id.add(child_node.id)

    def delete_all_children(self):
        self._child_prefix_dict = {}
        self._children_id = set()

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, new_position):
        self._position = new_position

    @property
    def child_prefix_dict(self):
        return self._child_prefix_dict


    @child_prefix_dict.setter
    def child_prefix_dict(self, dict_):
        self._child_prefix_dict = dict_

    @property
    def children_id(self):
        return self._children_id

    @children_id.setter
    def children_id(self, children_id_):
        self._children_id = children_id_

    @property
    def child_prefix(self):
        return self._child_prefix_dict

    @property
    def suffix_link(self):
        '''Suffix link getter
        '''
        return self._suffix_link

    @suffix_link.setter
    def suffix_link(self, node_id):
        '''Suffix link setter
        '''
        self._suffix_link = node_id

    @property
    def data(self):
        '''Node data getter
        '''
        return self._data

    @data.setter
    def data(self, data_):
        '''Node data setter
        '''
        self._data = data_

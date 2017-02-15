import uuid
import copy

from py.utils.tree import TreeNode, Tree


class SuffixTree(Tree):
    """This class is used to construct suffix tree
    """

    def __init__(self, dataset, sentence_delimiter='\n'):
        Tree.__init__(self)
        root_node = SuffixTreeNode(data=[])
        self.add_node(root_node)
        self._root = self.get_root()
        self._dataset = dataset
        self._sentence_delimiter = sentence_delimiter
        self._leaves = set()
        self._status = None

    def construct(self, data, sentence_delimiter='\n'):
        """Construct suffix tree
        """
        lines = data.split(sentence_delimiter)
        for line in lines:
            words = line.split()
            words.append('$')
            suffixes = [words[i:] for i in range(len(words))]
            for suffix in suffixes:
                # insert_suffix(suffix)
                pass

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

    def add_to_root(self, status):

    def add_sentence(self, words):
        # current_position = -1
        status = {'active_node': self._root,
                  'active_edge': None,
                  'active_length': 0
                  }
        leaves = set()
        remainder = 0
        previous_node_added = None

        for i, word in enumerate(words):
            remainder += 1
            active_node = status['active_node']
            for leaf in leaves:
                leaf_node = self.get_node(leaf)
                leaf_node.append_suffix(word)

            if active_node.is_root():
                if status['active_edge'] is None:
                    if not active_node.children_prefix(word):
                        node = SuffixTreeNode(data=[word])
                        self.add_node(node, parent=self.root_id)
                        leaves.add(node.id)
                        remainder -= 1
                    else:
                        edge_node_id = active_node.children_prefix(word)
                        edge_node = self.get_node(edge_node_id)
                        status['active_edge'] = edge_node
                        status['active_length'] += 1

                else:
                    active_edge_words = status['active_edge'].data
                    if active_edge_words[status['active_length']] != word:
                        for j in range(remainder):
                            new_node = SuffixTreeNode(data=[word, ])
                            if status['active_edge'] is not None:
                                # split the node
                                status['active_node'].remove_child(status['active_edge'])
                                other_node = copy.deepcopy(status['active_edge'])
                                # other_node.parent_id = status['active_edge'].id
                                other_node.data = status['active_edge'].data[status['active_length']:]
                                status['active_edge'].data = status['active_edge'].data[:status['active_length']]
                                other_node.clean_suffix_link()
                                # if status['active_edge'].id in leaves:
                                #     leaves.remove(status['active_edge'].id)
                                #     leaves.add(other_node.id)
                                status['active_edge'].delete_all_children()
                                status['active_edge'].id = uuid.uuid1().hex
                                other_node.parent_id = status['active_edge'].id

                                self._dict[other_node.id] = other_node
                                status['active_edge'].add_child(other_node)

                                self.add_node(status['active_edge'], parent=status['active_node'])
                                if previous_node_added is not None:
                                    previous_node_added.suffix_link = status['active_edge'].id
                                previous_node_added = status['active_edge']
                                # self.add_node(other_node, parent=status['active_edge'])
                                self.add_node(new_node, parent=status['active_edge'])
                                leaves.add(new_node.id)
                                remainder -= 1
                                if status['active_node'].children_prefix(words[i - remainder + 1]):
                                    status['active_edge'] = self.get_node(
                                        status['active_node'].children_prefix(words[i - remainder + 1]))
                                else:
                                    status['active_edge'] = None

                                status['active_length'] -= 1
                            else:
                                assert remainder == 1
                                node = SuffixTreeNode(data=[word])
                                self.add_node(node, parent=status['active_node'])
                                leaves.add(node.id)
                                remainder -= 1

                    else:
                        status['active_length'] += 1
                        if status['active_length'] == len(active_edge_words):
                            # TODO
                            # active_node changed
                            status['active_node'] = status['active_edge']
                            status['active_edge'] = None
                            status['active_length'] = 0
            else:
                if status['active_edge'] is None:
                    if not active_node.children_prefix(word):
                        while status['active_node']:
                            node = SuffixTreeNode(data=[word])
                            self.add_node(node, parent=status['active_node'].id)
                            leaves.add(node.id)
                            remainder -= 1
                            status['active_node'] = self.get_node(status['active_node'].suffix_link)
                        status['active_node'] = self.get_root()
                        node = SuffixTreeNode(data=[word])
                        self.add_node(node, parent=status['active_node'].id)
                        leaves.add(node.id)
                        remainder -= 1
                    else:
                        edge_node_id = active_node.children_prefix(word)
                        edge_node = self.get_node(edge_node_id)
                        status['active_edge'] = edge_node
                        status['active_length'] += 1
                else:
                    active_edge_words = status['active_edge'].data
                    if active_edge_words[status['active_length']] != word:
                        while not status['active_node'].is_root():
                            new_node = SuffixTreeNode(data=[word, ])
                            # split the node
                            status['active_node'].remove_child(status['active_edge'])
                            other_node = copy.deepcopy(status['active_edge'])
                            # other_node.parent_id = status['active_edge'].id
                            other_node.data = status['active_edge'].data[status['active_length']:]
                            status['active_edge'].data = status['active_edge'].data[:status['active_length']]
                            other_node.clean_suffix_link()
                            # if status['active_edge'].id in leaves:
                            #     leaves.remove(status['active_edge'].id)
                            #     leaves.add(other_node.id)
                            status['active_edge'].delete_all_children()
                            status['active_edge'].id = uuid.uuid1().hex
                            other_node.parent_id = status['active_edge'].id
                            self._dict[other_node.id] = other_node

                            status['active_edge'].add_child(other_node)

                            self.add_node(status['active_edge'], parent=status['active_node'])
                            if previous_node_added is not None:
                                previous_node_added.suffix_link = status['active_edge'].id
                            previous_node_added = status['active_edge']
                            self.add_node(new_node, parent=status['active_edge'])
                            leaves.add(new_node.id)
                            remainder -= 1
                            if status['active_node'] == self.get_root():
                                status['active_node'] = None
                            else:
                                status['active_node'] = self.get_node(
                                    status['active_node'].suffix_link) if \
                                    status['active_node'].suffix_link else self.get_root()
                                new_edge_id = status['active_node'].children_prefix(
                                    status['active_edge'].first_element())
                                status['active_edge'] = self.get_node(new_edge_id) if new_edge_id else None
                        # assert remainder == 1
                        node = SuffixTreeNode(data=[word])
                        self.add_node(node, parent=status['active_node'])
                        leaves.add(node.id)
                        remainder -= 1
                    else:
                        status['active_length'] += 1
                        if status['active_length'] == len(active_edge_words):
                            # TODO
                            # active_node changed
                            status['active_node'] = status['active_edge']
                            status['active_edge'] = None
                            status['active_length'] = 0
                        else:
                            status['active_length'] += 1
            tmp_str = self.tree_str(self.get_root())
            previous_node_added = None

            # def process(self, index, status, remainder):


class SuffixTreeNode(TreeNode):
    '''Suffix tree node class, inherit from TreeNode
    '''

    def __init__(self, data, id_=None, parent_id=None):
        if id_ is None:
            id_ = uuid.uuid1().hex
        TreeNode.__init__(self, id_=id_, parent_id=parent_id)
        self._data = data
        self._child_prefix_dict = {}
        self._suffix_link = None

    def first_element(self):
        '''Return the first element of node data
        '''
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

    def clean_suffix_link(self):
        self._suffix_link = None

    def change_child(self, old_child, new_child):
        if old_child == new_child:
            return
        self._child_prefix_dict[new_child.child_node.first_element()] = new_child.id
        self._children_id.remove(old_child.id)
        self._children_id.add(new_child.id)

    def remove_child(self, child):
        self._child_prefix_dict[child.first_element()] = None
        self._children_id.remove(child.id)




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

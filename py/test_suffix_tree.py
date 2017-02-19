from py.data_preprocessing.suffix import XTree
import uuid


# for backup
def add_to_root(self, i, words):
    word = words[i]
    if self._status['active_edge'] is None:
        if not self._status['active_node'].children_prefix(word):
            node = SuffixTreeNode(data=[word])
            self.add_node(node, parent=self._status['active_node'])
            self._leaves.add(node.id)
            self._remainder -= 1
        else:
            edge_node_id = self._status['active_node'].children_prefix(word)
            edge_node = self.get_node(edge_node_id)
            self._status['active_edge'] = edge_node
            self._status['active_length'] += 1

    else:
        # if active_edge_words[self._status['active_length']] != word:
        # for j in range(self._remainder):
        while self._status['active_edge'] and self._remainder > 0 and \
                        self._status['active_edge'].data[self._status['active_length']] != word:
            new_node = SuffixTreeNode(data=[word, ])
            # if self._status['active_edge'] is not None:
            # split the node
            self._status['active_node'].remove_child(self._status['active_edge'])
            other_node = copy.deepcopy(self._status['active_edge'])
            other_node.data = self._status['active_edge'].data[self._status['active_length']:]
            self._status['active_edge'].data = self._status['active_edge'].data[
                                               :self._status['active_length']]
            other_node.clean_suffix_link()
            self._status['active_edge'].delete_all_children()
            self._status['active_edge'].id = uuid.uuid1().hex
            other_node.parent_id = self._status['active_edge'].id

            self._dict[other_node.id] = other_node
            self._status['active_edge'].add_child(other_node)

            self.add_node(self._status['active_edge'], parent=self._status['active_node'])
            if self._previous_node_added is not None:
                self._previous_node_added.suffix_link = self._status['active_edge'].id
                self._reverse_suffix_links[self._status['active_edge']] = \
                    self._previous_node_added.suffix_link
            self._previous_node_added = self._status['active_edge']
            self.add_node(new_node, parent=self._status['active_edge'])
            self._leaves.add(new_node.id)
            self._remainder -= 1
            if self._status['active_node'].children_prefix(words[i - self._remainder + 1]):
                self._status['active_edge'] = self.get_node(
                    self._status['active_node'].children_prefix(words[i - self._remainder + 1]))
            else:
                self._status['active_edge'] = None

            self._status['active_length'] -= 1

        if self._remainder == 0:
            return
        elif self._status['active_edge'] is None:
            assert self._remainder == 1
            node = SuffixTreeNode(data=[word])
            self.add_node(node, parent=self._status['active_node'])
            self._leaves.add(node.id)
            self._remainder -= 1
        else:
            self._status['active_length'] += 1
            if self._status['active_length'] == len(self._status['active_edge'].data):
                # TODO
                # active_node changed
                self._status['active_node'] = self._status['active_edge']
                self._status['active_edge'] = None
                self._status['active_length'] = 0

def add_sentence(self, words):
    self._status = {'active_node': self._root,
                    'active_edge': None,
                    'active_length': 0
                    }
    self._leaves = set()
    self._remainder = 0
    self._previous_node_added = None

    for i, word in enumerate(words):
        for leaf in self._leaves:
            leaf_node = self.get_node(leaf)
            leaf_node.append_suffix(word)
        self._remainder += 1
        if self._status['active_node'].is_root():
            self.add_to_root(i, words)
        else:
            if self._status['active_edge'] is None:
                if not self._status['active_node'].children_prefix(word):
                    while self._status['active_node']:
                        node = SuffixTreeNode(data=[word])
                        self.add_node(node, parent=self._status['active_node'].id)
                        self._leaves.add(node.id)
                        self._remainder -= 1
                        self._status['active_node'] = self.get_node(self._status['active_node'].suffix_link) if \
                            self._status['active_node'].suffix_link else None
                    self._status['active_node'] = self.get_root()
                    node = SuffixTreeNode(data=[word])
                    self.add_node(node, parent=self._status['active_node'].id)
                    self._leaves.add(node.id)
                    self._remainder -= 1
                else:
                    edge_node_id = self._status['active_node'].children_prefix(words[i])
                    edge_node = self.get_node(edge_node_id)
                    self._status['active_edge'] = edge_node
                    self._status['active_length'] += 1
                    if self._status['active_length'] == len(self._status['active_edge'].data):
                        # TODO
                        # active_node changed
                        self._status['active_node'] = self._status['active_edge']
                        self._status['active_edge'] = None
                        self._status['active_length'] = 0

            else:
                active_edge_words = self._status['active_edge'].data
                if active_edge_words[self._status['active_length']] != word:
                    while not self._status['active_node'].is_root():
                        new_node = SuffixTreeNode(data=[word, ])
                        # split the node
                        self._status['active_node'].remove_child(self._status['active_edge'])
                        other_node = copy.deepcopy(self._status['active_edge'])
                        other_node.data = self._status['active_edge'].data[self._status['active_length']:]
                        self._status['active_edge'].data = \
                            self._status['active_edge'].data[:self._status['active_length']]
                        other_node.clean_suffix_link()
                        self._status['active_edge'].delete_all_children()
                        self._status['active_edge'].id = uuid.uuid1().hex
                        if self._reverse_suffix_links.get(other_node.id):
                            self.get_node(self._reverse_suffix_links[other_node.id]).suffix_link = \
                                self._status['active_edge'].id
                            self._reverse_suffix_links[self._status['active_edge'].id] = \
                                self._reverse_suffix_links[other_node.id]
                            self._reverse_suffix_links[other_node.id] = None
                        other_node.parent_id = self._status['active_edge'].id
                        self._dict[other_node.id] = other_node

                        self._status['active_edge'].add_child(other_node)

                        self.add_node(self._status['active_edge'], parent=self._status['active_node'])
                        if self._previous_node_added is not None:
                            self._previous_node_added.suffix_link = self._status['active_edge'].id
                            self._reverse_suffix_links[self._status['active_edge']] = \
                                self._previous_node_added.suffix_link
                        self._previous_node_added = self._status['active_edge']
                        self.add_node(new_node, parent=self._status['active_edge'])
                        self._leaves.add(new_node.id)
                        self._remainder -= 1
                        if self._status['active_node'] == self.get_root():
                            self._status['active_node'] = None
                        else:
                            self._status['active_node'] = self.get_node(
                                self._status['active_node'].suffix_link) if \
                                self._status['active_node'].suffix_link else self.get_root()
                            new_edge_id = self._status['active_node'].children_prefix(
                                self._status['active_edge'].first_element())
                            self._status['active_edge'] = self.get_node(new_edge_id) if new_edge_id else None
                    self.add_to_root(i, words)
                else:
                    self._status['active_length'] += 1
                    if self._status['active_length'] == len(active_edge_words):
                        # TODO
                        # active_node changed
                        self._status['active_node'] = self._status['active_edge']
                        self._status['active_edge'] = None
                        self._status['active_length'] = 0
        self._tmp_str = self.tree_str(self.get_root())
        self._previous_node_added = None

    return self._tmp_str

def add_sentence_2(self, words):
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
                        status['active_length'] += 1
        tmp_str = self.tree_str(self.get_root())
        previous_node_added = None

        # def process(self, index, status, remainder):

def construct_suffix(self, sentence_delimiter='\n'):
        """Construct suffix tree
        """
        lines = self._data.split(sentence_delimiter)
        for i, line in enumerate(lines):
            words = line.split()
            words.append('<eos_' + str(i) + '>')
            result = self.add_sentence(words)
        print(result)

def construct_prefix(self, sentence_delimiter='\n'):
        lines = self._data.split(sentence_delimiter)
        for i, line in enumerate(lines):
            words = line.split()
            words = list(reversed(words))
            words.append('<bos_' + str(i) + '>')
            result = self.add_sentence(words)
        print(result)

if __name__ == '__main__':
    test_data = 'a b c a b x a b c d\na b c d e f g\na b e g i a\ne c d a c b i'
    xtree = XTree(test_data)
    prefix_result, suffix_result = xtree.fetch_prefix_suffix('a b')
    print(prefix_result)
    print(suffix_result)



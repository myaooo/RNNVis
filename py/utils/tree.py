"""
A simple implementation of tree structure
"""

import uuid
from copy import deepcopy


class TreeNode(object):
    """
    Node of Tree, for the sake of data persistence, all the "pointers" are implemented as an id,
    which are managed by a dict in the tree.
    """

    def __init__(self, parent_id=None, id_=None):
        self._parent_id = parent_id
        self._children_id = []
        self._id = None
        self._set_id(id_)

    def _set_id(self, id_):
        if id_ is None:
            self._id = uuid.uuid1().hex
        else:
            self._id = id_

    @property
    def parent_id(self):
        return self._parent_id

    @parent_id.setter
    def parent_id(self, parent_id):
        self._parent_id = parent_id
        # else:
        #     self._parent = parent_id
        # else:
        #     raise TypeError("parent of of a TreeNode must be an instance of TreeNode!")

    @property
    def children_id(self):
        return self._children_id

    @property
    def id(self):
        return self._id

    def is_leaf(self):
        return len(self.children_id) == 0

    def is_root(self):
        return self.parent_id is None

    def as_dict(self):
        node_dict = deepcopy(self.__dict__)
        node_dict['parent_id'] = node_dict.pop('_parent_id')
        node_dict['children_id'] = node_dict.pop('_children_id')
        id_ = node_dict.pop('_id')  # remove the _id for efficiency
        return {id_: node_dict}


class Tree(object):

    def __init__(self):
        self.root_id = None
        self._dict = {}

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
            parent.children_id.append(node.id)
            node.parent_id = parent.id
        self._dict[node.id] = node

    def get_root(self):
        return self.get_node_by_id(self.root_id)

    def get_children(self, node_or_id, rank=None):
        node = self.get_node(node_or_id)
        if rank is None:
            return [self.get_node_by_id(id_) for id_ in node.children_id]

    def has_node_id(self, node_id):
        return node_id in self._dict

    def has_node(self, node_or_id):
        if isinstance(node_or_id, str):
            return self.has_node_id(node_or_id)
        elif isinstance(node_or_id, TreeNode):
            return self.has_node_id(node_or_id.id)
        else:
            raise TypeError("node_or_id should be a str or a TreeNode, but it's of type {:s}".format(type(node_or_id)))

    def get_node_by_id(self, node_id):
        if self.has_node_id(node_id):
            return self._dict[node_id]
        return None

    def get_node(self, node_or_id):
        if isinstance(node_or_id, str):
            return self.get_node_by_id(node_or_id)
        elif isinstance(node_or_id, TreeNode):
            return self.get_node_by_id(node_or_id.id)
        else:
            raise TypeError("node_or_id should be a str or a TreeNode, but it's of type {:s}".format(type(node_or_id)))

    def as_dict(self):
        """
        Return a dict object with the following format:
        {
            'root_id': 'XXidXX',
            'nodes':
            {'XXidXX': {'parent_id': 'XXidXX', 'children_id': ['XXidXX', 'XXidXX', ...], 'data': ...},
             ...
            }
        }
        :return: a dict object storing the complete structure of the tree
        """
        tree_dict = {'root_id': self.root_id}
        nodes = {}
        for node in self._dict.values():
            # print(node.as_dict())
            nodes.update(node.as_dict())
        tree_dict['nodes'] = nodes
        return tree_dict

    def nodes(self):
        return self._dict.values()

    def __getitem__(self, key):
        try:
            return self._dict[key]
        except:
            raise KeyError("{:s} is not in tree".format(key))

    def __setitem__(self, key, node):
        self._dict[key] = node

    def __len__(self):
        return len(self._dict)

    def __contains__(self, node):
        if isinstance(node, TreeNode):
            return node.id in self._dict
        raise TypeError("node should be of type TreeNode but of type {:s}".format(type(node)))


if __name__ == '__main__':

    tree = Tree()
    node1 = TreeNode()
    print(node1.id)
    tree.add_node(node1)
    node2 = TreeNode()
    print(node2.id)
    tree.add_node(node2, node1)
    node3 = TreeNode()
    tree.add_node(node3, node1)
    print(tree.as_dict())

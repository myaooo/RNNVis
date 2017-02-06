class PrefixTree():
    """This class is used to construct prefix tree (Trie tree)"""

    def __init__(self, filename):
        self.filename = filename
    
    def construct(self):
        """Construct prefix tree"""
        pass

class Node():
    """This is node object in the prefix tree and suffix tree"""

    def __init__(self, content):
        self._content = content
        self._next = None
        self._child = None
    
    @property
    def content(self):
        """Getter"""
        return self._content
    
    @content.setter
    def content(self, value):
        """Setter"""
        self._content = value

    @property
    def next(self):
        """Getter"""
        return self._next
    
    @next.setter
    def next(self, value):
        """Setter"""
        self._next = value

    @property
    def child(self):
        """Getter"""
        return self._child
    
    @child.setter
    def child(self, value):
        """Setter"""
        self._child = value


class Node:

    def __init__(self, left = None, right = None, leaf = False, value = None):
        self.left = left
        self.right = right
        self.leaf = leaf
        self.value = value


    def set_as_internal_node(self, left, right):
        if not self.leaf:   
            self.leaf = False
            self.left = left
            self.right = right
            
            
    def is_leaf(self):
        return self.leaf
    
    def get_value(self):
        return self.value
    

from node import Node
from collections import Counter

class Decision_tree:
    splitting_criteria = None
    max_depth = None
    max_leaves = None
    root = None
    
    def __init__(self, splitting_criteria = 'gini', max_depth = 3, max_leaves = 5):
        self.splitting_criteria = splitting_criteria
        self.max_depth = max_depth
        self.max_leaves = max_leaves
 
        
    #takes the data and the attribute and compute the information gain 
    def _gain(self, X, a):
        pass
      
    def fit(self, X, y):
        #grow root node
        self._init_root(y)
        
        #for each leaf until stop
        self._replace_and_split(X,y,self.root)
   
    def _init_root(self, y):
        counter = Counter(y['class'])
        most_common = counter.most_common(1)[0][0]
        self.root = Node(leaf = True, value = most_common)
        
    def _replace_and_split(self,X,y,leaf):
        left = Node(leaf = True)
        right = Node(leaf = True)
        leaf.set_as_internal_node(left,right) #change its status in order not to change the pointer, hence preserving the tree


         
    def predict(self, X):
        if self.root is None:
            raise Exception('Tree not fitted')
        
        results = []
        for i in range(len(X)):
            if self.root.is_leaf:
                results.append(self.root.value)
            else:
                pass
        return results
        
    
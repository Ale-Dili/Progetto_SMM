from node import Node
from collections import Counter
import numpy as np
from math import log2
import pandas as pd

class Decision_tree:
    
    def __init__(self, splitting_criteria = 'gini', max_depth = 3, max_leaves = 5, min_samples_split=2):
        self.splitting_criteria = splitting_criteria
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_samples_split = min_samples_split
        self.mistakes = 0
        self.n_bins = 20
        self.thresholds = {}
      
    ###----funzioni helper-------
    def _is_continuous(self, X_column):
        return pd.api.types.is_numeric_dtype(X_column)
    
    def _create_thresholds(self,X):
        for colname in X.columns:
            if self._is_continuous(X[colname]):
                _, bin_edges = pd.cut(X[colname], bins=self.n_bins, retbins=True)
                self.thresholds[colname] = bin_edges
            else:
                self.thresholds[colname] = X[colname].unique()
            
    ###--------------------------     
        
    ###----funzioni public---------
    def fit(self, X, y):
        self._create_thresholds(X)
        self._grow_tree(X,y,0)
        
        
    def predict(self, X):
        pass
    ###--------------------------     

        
    ###----funzioni albero---------
    def _grow_tree(self,X,y,depth):
        pass
        
    ###--------------------------
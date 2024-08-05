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
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        print(best_feature, best_threshold, best_gain)


        

    def _best_split(self, X, y):
        best_gain = 0
        best_feature, best_threshold = None, None
        
        for colname in X.columns:
            X_column = X[colname]
            idxs_no_nan = X_column.dropna().index
            continuous = self._is_continuous(X_column)
            for threshold in self.thresholds[colname]:
                gain = self._information_gain(y.loc[idxs_no_nan], X_column.loc[idxs_no_nan],threshold, continuous)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = colname
                    best_threshold = threshold
        return best_feature, best_threshold, best_gain


    #Nan not present in here
    def _information_gain(self, y, X_column, threshold, continuous):
        parent_purity = self._criterion(y)
        
        left_idxs, right_idxs = self._split(X_column, threshold, continuous)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(X_column)
        n_left = len(left_idxs)
        n_right = len(right_idxs)
        
        left_purity = self._criterion(y.loc[left_idxs])
        right_purity = self._criterion(y.loc[right_idxs])
                
        children_purity = (n_left / n) * left_purity + (n_right / n) * right_purity
        gain = parent_purity - children_purity
        return gain

        
        
    def _split(self, X_column, threshold, continuous):
        if continuous:
            left_idxs = np.argwhere(X_column > threshold).flatten()
            right_idxs = np.argwhere(X_column <= threshold).flatten()
        else:
            left_idxs = np.argwhere(X_column == threshold).flatten()
            right_idxs = np.argwhere(X_column != threshold).flatten()
    
        left_idxs = X_column.index[left_idxs]
        right_idxs = X_column.index[right_idxs]
        
        return left_idxs, right_idxs
               
        
    def _criterion(self, y):
        #compute probability of positive class
        counts = y['class'].value_counts()
        num_pos = counts.get(1,0)
        num_neg = counts.get(0,0)
        if num_neg == 0 or num_pos == 0:
            return 0
        
        p_pos = num_pos/(num_neg+num_pos)
        if self.splitting_criteria == 'gini':
            return self._gini(p_pos)
        elif self.splitting_criteria == 'entropy':
            return self._entropy(p_pos)
         
    
    def _entropy(self, p):
        entropy = -p*log2(p)-(1-p)*log2(1-p)
        return entropy
     
    
    def _gini(self, p):
        gini = 2*p*(1-p)
        return gini


    ###--------------------------
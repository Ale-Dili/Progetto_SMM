from node import Node
from collections import Counter
import numpy as np
from math import log2
import pandas as pd
from numpy import sqrt

class Decision_tree:
    
    def __init__(self, splitting_criteria='gini', max_depth=3, min_samples_split=2, min_impurity_decrease=0.001):
        self.splitting_criteria = splitting_criteria
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.n_bins = 50
        self.thresholds = {}
        self.root = None
        self.mistakes = 0  # Initialize mistakes here
      
    ###----funzioni helper-------
    def _is_continuous(self, X_column):
        return pd.api.types.is_numeric_dtype(X_column)
    
    def _create_thresholds(self, X):
        for colname in X.columns:
            if self._is_continuous(X[colname]):
                _, bin_edges = pd.cut(X[colname], bins=self.n_bins, retbins=True)
                self.thresholds[colname] = bin_edges
            else:
                self.thresholds[colname] = X[colname].unique()
            
    ###--------------------------     
        
    ###----funzioni public---------
    def fit(self, X, y):
        self.mistakes = 0
        self._create_thresholds(X)
        self.root = self._grow_tree(X, y, 0)
        t_e = self._compute_training_error(X)
        return t_e
        
    def predict(self, X):
        if self.root is None:
            raise Exception('Tree not fitted')
        predictions = []
        for _, row in X.iterrows():  # Itera su ogni riga del DataFrame X
            prediction = self._traverse_tree(row, self.root)
            predictions.append(prediction)
        return predictions

    ###--------------------------     

        
    ###----funzioni albero---------
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if pd.isna(x[node.feature]):
             return self._traverse_tree(x, node.left)

 
        if isinstance(x[node.feature], float):
            if x[node.feature] > node.threshold:
                return self._traverse_tree(x, node.right)
            else:
                return self._traverse_tree(x, node.left)
        else:
            if x[node.feature] == node.threshold:
                return self._traverse_tree(x, node.right)
            else:
                return self._traverse_tree(x, node.left)
  
    
    def _compute_training_error(self, X):
        return self.mistakes / X.shape[0]
  
    
    def _grow_tree(self, X, y, depth):
        if len(y['class']) < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            self.mistakes += y.size - y.value_counts()[leaf_value]
            return Node(value=leaf_value)
        
        # Stopping criteria
        if len(set(y['class'])) == 1:
            leaf_value = self._most_common_label(y)
            self.mistakes += y.size - y.value_counts()[leaf_value]
            return Node(value=leaf_value)
        
        if depth >= self.max_depth:
            leaf_value = self._most_common_label(y)
            self.mistakes += y.size - y.value_counts()[leaf_value]
            return Node(value=leaf_value)
        
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_feature is None or best_gain < self.min_impurity_decrease:
            leaf_value = self._most_common_label(y)
            self.mistakes += y.size - y.value_counts()[leaf_value]
            return Node(value=leaf_value)
        
        X_column = X[best_feature]
        
        left_idxs, right_idxs = self._split(X_column, best_threshold)
        
        left_child = self._grow_tree(X.loc[left_idxs], y.loc[left_idxs], depth + 1)
        right_child = self._grow_tree(X.loc[right_idxs], y.loc[right_idxs], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
 
    
    def _most_common_label(self, y):
        return y['class'].mode()[0]

        
    def _best_split(self, X, y):
        best_gain = 0
        best_feature, best_threshold = None, None
        
        for colname in X.columns:
            X_column = X[colname]
            continuous = self._is_continuous(X_column)
            for threshold in self.thresholds[colname]:
                gain = self._information_gain(y, X_column, threshold, continuous)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = colname
                    best_threshold = threshold
        return best_feature, best_threshold, best_gain


    def _information_gain(self, y, X_column, threshold, continuous):
        parent_purity = self._criterion(y)
        
        if parent_purity == 0:  # Node is already pure
            return 0
        
        left_idxs, right_idxs = self._split(X_column, threshold)
        
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


    def _split(self, X_column, threshold):
        left_idxs = X_column.index[X_column <= threshold] if self._is_continuous(X_column) else X_column.index[X_column != threshold]
        right_idxs = X_column.index[X_column > threshold] if self._is_continuous(X_column) else X_column.index[X_column == threshold]
        
        return left_idxs, right_idxs
                
                
    def _criterion(self, y):
        # Compute probability of positive class
        counts = y['class'].value_counts()
        num_pos = counts.get(1, 0)
        num_neg = counts.get(0, 0)
        if num_neg == 0 or num_pos == 0:
            return 0
        
        p_pos = num_pos / (num_neg + num_pos)
        if self.splitting_criteria == 'gini':
            return self._gini(p_pos)
        elif self.splitting_criteria == 'entropy':
            return self._entropy(p_pos)
        elif self.splitting_criteria =='std_dev':
            return self._std_dev(p_pos)
    
    
    def _entropy(self, p):
        if p == 0 or p == 1:
            return 0
        return -(p/2) * log2(p) - ((1 - p)/2) * log2(1 - p)

    def _gini(self, p):
        return 2 * p * (1 - p)

    def _std_dev(self, p):
        return sqrt(p*(1-p))
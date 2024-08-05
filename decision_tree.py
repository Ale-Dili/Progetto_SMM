from node import Node
from collections import Counter
import numpy as np
from math import log2
import pandas as pd

class Decision_tree:
    splitting_criteria = None
    max_depth = None
    max_leaves = None
    root = None
    leaves = []
    temp_max_iter = 4
    
    def __init__(self, splitting_criteria = 'gini', max_depth = 3, max_leaves = 5, min_samples_split=2):
        self.splitting_criteria = splitting_criteria
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_samples_split = min_samples_split
        self.mistakes = 0
        self.n_bins = 30
     
    def get_mistake(self):
        return self.mistakes
    
    def _bin_continuos(self, X):
        #TODO fare check continua
        for colname in X.columns:
            if colname in ['cap-diameter', 'stem-height', 'stem-width']:
                X[colname] = pd.cut(X[colname], bins=self.n_bins)
        return X
      
    def fit(self, X, y):
        X = self._bin_continuos(X)
        self.root = self._grow_tree(X, y, depth=0)
        
    def _grow_tree(self, X, y, depth):
        num_samples, num_features = X.shape

        if len(set(y['class']))==1:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
    
        if depth >= self.max_depth:
            try:
                leaf_value = self._most_common_label(y)
            except:
                print(y)
            
            self.mistakes+= y.size - y.value_counts()[leaf_value]
            return Node(value=leaf_value)
        
        if num_samples < self.min_samples_split:
            self.mistakes+= y.size - y.value_counts()[leaf_value]
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        #best_feature is index of column
        best_feature, best_threshold = self._best_split(X, y, num_samples, num_features)
        
        print(X.columns[best_feature], best_threshold)
        
        X_column = X.iloc[:, best_feature]
        idxs_no_nan = X_column.dropna().index
        idxs_nan= X_column[X_column.isna()].index
           
        is_continuous = self._is_feature_continuous(X_column)
        
        #TODO
        if is_continuous:
            pass
        else:
            left_idxs, right_idxs = self._split(X_column.loc[idxs_no_nan], best_threshold, False)
            print(len(left_idxs), len(right_idxs))
        
        if (len(left_idxs)>len(right_idxs)):
            left_idxs = np.concatenate((left_idxs, idxs_nan))
        else:
            right_idxs = np.concatenate((right_idxs, idxs_nan))
    
        left_child = self._grow_tree(X.loc[left_idxs], y.loc[left_idxs], depth+1)
        right_child = self._grow_tree(X.loc[right_idxs], y.loc[right_idxs], depth+1)
        return Node(feature= best_feature, threshold= best_threshold, left = left_child, right=right_child )
    
        #find best attribute
         
        
        
    def _best_split(self,X,y, num_samples, num_features):
        best_gain = -float("inf")
        best_feature, best_threshold = None, None

        for feature_idx in range(num_features):
            X_column = X.iloc[:, feature_idx]
            idxs_no_nan = X_column.dropna().index
            idxs_nan= X_column[X_column.isna()].index
            
            #TODO
            if X.columns[feature_idx] in ['cap-diameter', 'stem-height', 'stem-width']:  # Continuous
                continue
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    gain = self._information_gain(y, X_column, threshold, continuous=True)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = threshold
            else:  # Categorical
                thresholds = np.unique(X_column.loc[idxs_no_nan])
                
                for threshold in thresholds:
                    #print(X.columns[feature_idx])
                   
                    gain = self._information_gain(y.loc[idxs_no_nan], X_column.loc[idxs_no_nan], threshold, continuous=False)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = threshold
        print(best_gain)
        return best_feature, best_threshold
     
    #TODO        
    def _is_feature_continuous(self, X_column):
        return False 
            
    def _information_gain(self, y, X_column, split_value, continuous ):
        parent_purity = self._criterion(y)
        left_idxs, right_idxs = self._split(X_column, split_value, continuous)
        
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
    
    
    
        
    def _split(self, X_column, split_value, continuous):
        if continuous:
            print(split_value)
            print(X_column.iloc[0])
            left_idxs = np.argwhere(X_column in split_value).flatten()
            right_idxs = np.argwhere(X_column not in split_value).flatten()
        else:
            left_idxs = np.argwhere(X_column > split_value).flatten()
            right_idxs = np.argwhere(X_column <=split_value).flatten()
        
        left_idxs = X_column.index[left_idxs]
        right_idxs = X_column.index[right_idxs]
        
        return left_idxs, right_idxs
    
    
    
    
    def _criterion(self, y):
        if self.splitting_criteria == 'gini':
            return self._gini(y)
        elif self.splitting_criteria == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError("Invalid splitting criteria")
    

    
    def _entropy(self, y):
        y = y.to_numpy().flatten()
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * log2(p) for p in ps if p > 0])
    
    def _gini(self, y):
        y = y.to_numpy().flatten()
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

   
    def _most_common_label(self, y):
        try:
            most_common = y['class'].mode()[0]
        except:
            print(y)
        return most_common 
         
    def predict(self, X):
        if self.root is None:
            raise Exception('Tree not fitted')
        predictions = []
        for _, row in X.iterrows():  # Itera su ogni riga del DataFrame X
            prediction = self._traverse_tree(row, self.root)
            predictions.append(prediction)
        return predictions


    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x.iloc[node.feature] == node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
        
    
from collections import Counter
import numpy as np
from math import log2
import pandas as pd
from numpy import sqrt
import decision_tree
from statistics import mode

class Random_Forest():
    def __init__(self, n_trees=10, seed=88):
        self.n_trees = n_trees
        self.seed = seed
        self.forest = []
        np.random.seed(seed)

    
    
    ###----funzioni public-------
    def fit(self, X, y):
        self.build_forest(X, y)
    
    def predict(self, X):
        predictions = []
        for tree in self.forest:
            predictions.append(tree.predict(X))
        
        predictions = np.array(predictions).T

        #majority voting
        final_predictions = []
        for i in range(len(predictions)):
            final_predictions.append(mode(predictions[i]))
            
        return final_predictions

            
    
    ###--------------------------
    
    ###----funzioni helper-------
    def build_forest(self, X, y):
        
        bootstrap_samples_X, bootstrap_samples_y = self._bootstrap_samples(X, y)        

        for i in range(self.n_trees):
            print(f'-- Training tree {i+1} --')
            random_seed = np.random.randint(0, 10001)
            print(f'   random seed: {random_seed}')
            tree = decision_tree.Decision_tree(splitting_criteria = 'entropy',is_forest=True, max_depth = 30, min_samples_split=1, min_impurity_decrease = 0.0005, seed=random_seed)
            t_e = tree.fit(bootstrap_samples_X[0], bootstrap_samples_y[0])
            print(f'   training error: {t_e}')
            self.forest.append(tree)

    
    def _bootstrap_samples(self, X, y):
        bootstrap_samples_X = []
        bootstrap_samples_y = []
        index_list = X.index.tolist()
        for _ in range(self.n_trees):
            indices = np.random.choice(index_list, size=len(index_list), replace=True)

            bootstrap_samples_X.append(X.loc[indices])
            bootstrap_samples_y.append(y.loc[indices])
        
        return bootstrap_samples_X, bootstrap_samples_y
    
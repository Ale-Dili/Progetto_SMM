from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold, KFold,LeaveOneOut)
import numpy as np
from ucimlrepo import fetch_ucirepo 
import random_forest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pickle

SEED = 44

def fix_label(y):
        y['class'] = y['class'].apply(lambda x: 1 if x == 'e' else 0)
        return y


##-------------Load dataset
mushroom = fetch_ucirepo(id=848) 
  
# data (as pandas dataframes) 
X = mushroom.data.features 
y = mushroom.data.targets 



X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, shuffle=True)


y = fix_label(y)
y_test = fix_label(y_test)

print('-- Dataset loaded --')
##-------------

forest = random_forest.Random_Forest(n_trees=29,seed=SEED)
oob_score = forest.fit(X, y)

print(f'OOB score: {oob_score}')



predictions = forest.predict(X_test)
#accuracy recall precision f1
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1: {f1}')

with open('trained_model_rf.pkl', 'wb') as f:  
    pickle.dump(forest, f) 
import importlib
import decision_tree
importlib.reload(decision_tree)  # Ricarica il modulo
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold, KFold,LeaveOneOut)
import numpy as np
from ucimlrepo import fetch_ucirepo 

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pickle

def fix_label(y):
        y['class'] = y['class'].apply(lambda x: 1 if x == 'e' else 0)
        return y


##-------------Load dataset
mushroom = fetch_ucirepo(id=848) 
  
# data (as pandas dataframes) 
X_tot = mushroom.data.features 
y_tot = mushroom.data.targets 



X, X_final_test, y, y_final_test = train_test_split(X_tot, y_tot, test_size=0.15, random_state=88, shuffle=True)


y = fix_label(y)
y_final_test = fix_label(y_final_test)

print('-- Dataset loaded --')
##-------------
param_grid = {
    'max_depth': [20,30,32], 
    'min_samples_split': [2,5,10],
    'criterion' : ['gini', 'entropy', 'sqrt_split'],
    'min_impurity_decrease' : [0.0005,0.001]
}

N_FOLDS = 5

#(max_depth, min_sample, criterion, min_impurity):{accuracy: int, precision: int, recall: int, f1: int}
metrics = {}

for max_depth in param_grid['max_depth']:
    for min_samples_split in param_grid['min_samples_split']:
        for criterion in param_grid['criterion']:
            for min_impurity_decrease in param_grid['min_impurity_decrease']:
                model = decision_tree.Decision_tree(splitting_criteria = criterion, max_depth = max_depth, min_samples_split=min_samples_split, min_impurity_decrease = min_impurity_decrease)
            
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=False)
                skf_accuracies=[]
                skf_precisions=[]
                skf_recalls = []
                skf_f1 = []
                print(f'-- TRAINING ::: max_dept= {max_depth} | min_samp = {min_samples_split} | criterion = {criterion} | min_impurity = {min_impurity_decrease} ::: --')
                count = 1
                for train_index, test_index in skf.split(X, y):
                    X_train = X.iloc[train_index]
                    y_train = y.iloc[train_index]
                    X_test = X.iloc[test_index]
                    y_test = y.iloc[test_index]
                    training_error = model.fit(X_train,y_train)
                    
                    print(f'    ({count}/{N_FOLDS}): training error: {training_error}')
                    
                    y_pred = model.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    
                    print(f'     ({count}/{N_FOLDS}): accuracy: {accuracy}')
                    
                    skf_accuracies.append(accuracy)
                    skf_precisions.append(precision)
                    skf_recalls.append(recall)
                    skf_f1.append(f1)
                    
                    count+=1
                    
                metrics[(max_depth, min_samples_split, criterion, min_impurity_decrease)] = {
                    'accuracy' : np.mean(skf_accuracies),
                    'precision' : np.mean(skf_precisions),
                    'recall' : np.mean(skf_recalls),
                    'f1': np.mean(skf_f1)
                }
                

                    
with open('metrics.pkl', 'wb') as f:  # open a text file
    pickle.dump(metrics, f) # serialize the list
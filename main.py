from ucimlrepo import fetch_ucirepo 
from decision_tree import Decision_tree
from sklearn.model_selection import train_test_split
from normalizer import Normalizer
import pandas as pd
import numpy as np



mushroom = fetch_ucirepo(id=848) 
  
attributes_type = mushroom.variables.type
# data (as pandas dataframes) 
X = mushroom.data.features 
y = mushroom.data.targets 


#Predict if mushroom is edible
#e = 1
#p = 0



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

normalizer = Normalizer(n_bins=4, normalization="minmax")

X_train = normalizer.normalize(X_train)
y_train = normalizer._fix_label(y_train)


#y_train = y_train.to_numpy().flatten()

# Decision tree classifier
dt = Decision_tree(splitting_criteria = 'entropy', max_depth = 18, max_leaves = 5, min_samples_split=2)

dt.fit(X_train, y_train)

print(dt.get_mistake())
#X_test = normalizer.normalize(X_test)

#results = dt.predict(X_test)
#print(results[:100])

from ucimlrepo import fetch_ucirepo 
from node import Node
from decision_tree import Decision_tree
from sklearn.model_selection import train_test_split


mushroom = fetch_ucirepo(id=848) 
  
# data (as pandas dataframes) 
X = mushroom.data.features 
y = mushroom.data.targets 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Decision tree classifier
dt = Decision_tree()

dt.fit(X_train, y_train)

#results = dt.predict(X_test)
#print(results[:10])
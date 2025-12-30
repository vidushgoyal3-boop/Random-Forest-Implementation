import RandomforestClassifier as rf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data  
y = data.target  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
obj=rf.RandomForestClassifier(n_trees=50, max_depth=10, n_features=None, min_samples_split=2)
obj.fit(X_train,y_train)
a=obj.predict(X_test)
accuracy=np.sum(a==y_test)/len(y_test)
print("Accuracy:",accuracy)

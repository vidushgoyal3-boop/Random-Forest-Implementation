import DecisiontreeClassifier as df
import numpy as np
from collections import Counter

class RandomForestClassifier:
    def __init__(self,*,min_samples_split=None,max_depth=None,n_features=None,n_trees=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.n_trees=n_trees
        self.trees=[]
    def fit(self,x,y):
        self.n_samples = x.shape[0]
        for i in range(self.n_trees):
            tree=df.DecisionTreeClassifier(max_depth=self.max_depth,n_features=self.n_features,min_samples_split=self.min_samples_split)
            bootstrap_idxs=np.random.choice(self.n_samples,self.n_samples,replace=True)
            x_,y_=x[bootstrap_idxs],y[bootstrap_idxs]                          
            tree.fit(x_,y_)
            self.trees.append(tree)
    def predict(self,x):
        predictions=np.array([tree.predict(x) for tree in self.trees])
        preds=np.swapaxes(predictions,0,1)
        predictions=np.array([Counter(pred).most_common(1)[0][0] for pred in preds])
        return predictions
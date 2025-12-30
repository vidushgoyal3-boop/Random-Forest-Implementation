import numpy as np
from collections import Counter

class Node:

    def __init__(self,*,feature=None,threshold=None,left=None,right=None,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value!=None


class DecisionTreeClassifier:

    def __init__(self,*,max_depth,n_features,min_samples_split):
        self.root=None
        self.max_depth = max_depth
        self.n_features = n_features
        self.min_samples_split = min_samples_split

    def fit(self,x,y):
        if(self.n_features==None):
            self.n_features=x.shape[1]
        self.root=self.tree(x,y)

    def tree(self,x,y,depth=0):
        labels=len(np.unique(y))
        n_fs=x.shape[1] #total no. of features
        samples=x.shape[0]
        if(depth>=self.max_depth or labels==1 or samples<=self.min_samples_split): # stopping conditions
            value=Counter(y).most_common(1)[0][0]
            return Node(value=value)
        features=np.random.choice(n_fs,self.n_features,replace=False)
        b_f,b_t=self.best_split(x,y,features)
        left_nodes, right_nodes = self.split(x[:,b_f],b_t)
        left = self.tree(x[left_nodes ,:], y[left_nodes], depth+1)
        right = self.tree(x[right_nodes, :], y[right_nodes], depth+1)
        return Node(feature=b_f,threshold=b_t, left=left,right= right)


    def best_split(self,x,y,features):
        b_f,b_t=None,None
        max_IG=-1
        for f in features:
            thresholds=np.unique(x[:,f])
            for t in thresholds:
                IG=self.information_gain(x[:,f],y,t)
                if(max_IG<IG):
                    b_f=f
                    b_t=t
                    max_IG=IG
        return b_f,b_t

    def information_gain(self,x_column,y,threshold):
        parent_size=len(y)
        e_parent=self.entropy(y)
        left_nodes,right_nodes=self.split(x_column,threshold)
        left_size=len(left_nodes)
        right_size=len(right_nodes)
        e_left=self.entropy(y[left_nodes])
        e_right=self.entropy(y[right_nodes])
        return e_parent-(left_size)/(parent_size)*e_left-(right_size)/(parent_size)*e_right

    def entropy(self,y):
        probs=np.bincount(y)/len(y)
        return -sum(p*np.log(p) for p in probs if p>0)

    def split(self,x_column,threshold):
        left_nodes=np.argwhere(x_column<=threshold).flatten()
        right_nodes=np.argwhere(x_column>threshold).flatten()
        return left_nodes,right_nodes

    def predict(self,x):
        return np.array([self.helper_predict(i,self.root) for i in x])
    
    def helper_predict(self,x_i,node):
        if(node.is_leaf()):
            return node.value
        if(x_i[node.feature]<=node.threshold):
            return self.helper_predict(x_i,node.left)
        else:
            return self.helper_predict(x_i,node.right)
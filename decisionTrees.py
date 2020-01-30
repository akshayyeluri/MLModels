import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
#from MLModels import utils as u

class DecisionTree(): 
    class Node():
        def __init__(self, value, depth):
            self.value=value
            self.depth=depth
            self.kids=None
            self.feature=None
            self.thresh=None
            
        def nPrint(self):
            print(f'Node at depth {self.depth}, value={self.value}')
            print(f'Feature: {self.feature}, Threshold: {self.thresh}, kids: {self.kids}')
            print()
            
            
    def __init__(self, maxDepth=None, impur=None, \
                 decision_func=lambda x: st.mode(x)[0][0]):
        '''
        Initialize a decision tree taking
        x input with d features
        '''
                                                                         
        self.maxDepth = maxDepth                                                        
        self.L = impur if impur else \
                 lambda Y: Y.shape[0] * st.entropy([Y.mean(), 1-Y.mean()])
        self.decision = decision_func
        self.root = None
        self.d = None # How many features do input points have?
        
    def tree_print(self, node=None):
        if not node:
            node=self.root
        node.nPrint()
        if node.kids:
            self.tree_print(node.kids[0])
            self.tree_print(node.kids[1])
        
    def split(self, node, X, Y):
        if node.depth >= self.maxDepth:
            return
        
        # Find best feature to split on and threshold
        # to split on
        bestFeat, bestThresh = None, None
        bestImp = self.L(Y)
        for feat in range(self.d):
            x = X[:, feat]
            inds = np.argsort(x)
            x, y = x[inds], Y[inds]
            for i, thresh in enumerate((x[:-1]+x[1:])/2):
                newImp = self.L(y[:(i+1)]) + self.L(y[i+1:])
                #print(feat, thresh, newImp)
                if newImp < bestImp:
                    bestFeat, bestThresh = feat, thresh
                    bestImp = newImp
        
        
        #print(bestFeat, bestThresh, bestImp)
        if bestFeat is None:
            return # Already perfectly classified
        
        # Set node data
        node.feature = bestFeat
        node.thresh = bestThresh
        
        # make node's kids
        inds = (X[:, bestFeat] < bestThresh)
        X_l, Y_l = X[inds], Y[inds]
        left_kid = self.Node(self.decision(Y_l), node.depth + 1)
        X_r, Y_r = X[~inds], Y[~inds]
        right_kid = self.Node(self.decision(Y_r), node.depth + 1)
        node.kids = [left_kid, right_kid]
        
        # recurse
        self.split(node.kids[0], X_l, Y_l)
        self.split(node.kids[1], X_r, Y_r) 
                                                                                                                                                      
    def learn(self, X, Y):                                                        
        self.root = self.Node(self.decision(Y), 0)
        self.d = X.shape[1]
        self.split(self.root, X, Y)
    
    def decide(self, node, X):
        if node.kids is None:
            return np.full(X.shape[0], node.value)
        Y = np.empty(X.shape[0])
        inds = (X[:, node.feature] < node.thresh)
        Y[inds] = self.decide(node.kids[0], X[inds])
        Y[~inds] = self.decide(node.kids[1], X[~inds])
        return Y
    
    def calculate(self, X):
        return self.decide(self.root, X)
    
    def get_partitions(self, tot_bounds=None):
        patches = [] 
        bounds = tot_bounds if tot_bounds else \
                            [-np.inf, np.inf] * self.d
        self.collect_partitions(self.root, bounds, patches)
        return np.array(patches)
        
    def collect_partitions(self, node, bounds, patches):
        #node.nPrint()
        if node.kids is None:
            patches.append(bounds + [node.value,])
            return
        bounds_l, bounds_r = bounds[:], bounds[:]
        bounds_r[0 + 2 * node.feature] = node.thresh
        bounds_l[1 + 2 * node.feature] = node.thresh
        self.collect_partitions(node.kids[0], bounds_l, patches)
        self.collect_partitions(node.kids[1], bounds_r, patches)    
            
    def quickPlot(self, X, Y, axis=None):
        if X.shape[1] != 2:
            raise ValueError('plotting requires 2d input')
        
        if axis is None:                                                        
            fig, ax = plt.subplots(1, 1)                                        
        else:                                                                   
            ax = axis
            
        bounds = [X[:,0].min(), X[:,0].max(), \
                  X[:,1].min(), X[:,1].max()]
        patches = self.get_partitions(bounds)
        for (x_min, x_max, y_min, y_max, v) in patches:
            ax.fill_between([x_min, x_max], [y_min, y_min], [y_max, y_max],\
                            alpha=0.3, color='b' if v else 'r')
        
        indsPos, indsNeg = (Y == 1), (Y == 0)                                  
        ax.plot(X[indsPos, -2], X[indsPos, -1], 'b+')                   
        ax.plot(X[indsNeg, -2], X[indsNeg, -1], 'r_')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
                                                                                
        if axis is None:                                                        
            return fig, ax

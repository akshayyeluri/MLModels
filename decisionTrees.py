import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import networkx as nx
#from MLModels import utils as u

class DecisionTree(): 
    class Node():
        def __init__(self, value, depth, iD):
            self.value=value
            self.depth=depth
            self.id = iD
            self.kids=None
            self.feature=None
            self.thresh=None
            
        def nPrint(self):
            print(f'Node at depth {self.depth}, value={self.value}')
            print(f'Feature: {self.feature}, Threshold: {self.thresh}, kids: {self.kids}')
            print()
            
            
    def __init__(self, impur=None, \
                 decision_func=lambda x: st.mode(x)[0][0]):
        '''
        Initialize a decision tree taking
        x input with d features
        '''                                                       
        self.L = impur if impur else \
                 lambda Y: Y.shape[0] * st.entropy([Y.mean(), 1-Y.mean()])
        self.decision = decision_func
        self.root = None
        self.totalNodes = 0
        self.d = None # How many features do input points have?
        # termination parameters
        self.maxDepth = None                    
        self.minPoints = None
        self.maxNodes = None
        
    def tree_print(self, node=None):
        if not node:
            node=self.root
        node.nPrint()
        if node.kids:
            self.tree_print(node.kids[0])
            self.tree_print(node.kids[1])
        
    def split(self, node, X, Y):
        if (node.depth >= self.maxDepth) or (self.totalNodes >= self.maxNodes - 2) :
            return
        
        # Find best feature to split on and threshold
        # to split on
        bestFeat, bestThresh = None, None
        bestImp = self.L(Y)
        for feat in range(self.d):
            x = X[:, feat]
            inds = np.argsort(x)
            x, y = x[inds], Y[inds]
            thresholds = (x[:-1]+x[1:])/2
            start, stop = self.minPoints, len(thresholds) - self.minPoints
            for i, thresh in enumerate(thresholds[start:stop]):
                cutoff = start + i + 1
                newImp = self.L(y[:cutoff]) + self.L(y[cutoff:])
                #print(feat, thresh, newImp)
                if newImp < bestImp:
                    bestFeat, bestThresh = feat, thresh
                    bestImp = newImp
        
        if bestFeat is None:
            return # Already perfectly classified, or other termination condition
        
        # Set node data
        node.feature = bestFeat
        node.thresh = bestThresh
        
        # make node's kids
        inds = (X[:, bestFeat] < bestThresh)
        X_l, Y_l = X[inds], Y[inds]
        left_kid = self.Node(self.decision(Y_l), node.depth + 1, self.totalNodes)
        self.totalNodes += 1
        X_r, Y_r = X[~inds], Y[~inds]
        right_kid = self.Node(self.decision(Y_r), node.depth + 1, self.totalNodes)
        node.kids = [left_kid, right_kid]
        self.totalNodes += 1
        
        # recurse
        self.split(node.kids[0], X_l, Y_l)
        self.split(node.kids[1], X_r, Y_r) 
                                                                                                                                                      
    def learn(self, X, Y, **conditions):
        # Get termination conditions
        self.maxDepth = conditions.get('maxDepth', np.inf)                           
        self.minPoints = conditions.get('minPoints', 0)
        self.maxNodes = conditions.get('maxNodes', np.inf)
        # Set root and number of features
        self.root = self.Node(self.decision(Y), 0, self.totalNodes)
        self.totalNodes += 1
        self.d = X.shape[1]
        # Recursive train
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
        
        inds = (Y == 1)
        ax.plot(X[inds, -2], X[inds, -1], 'b+')
        ax.plot(X[~inds, -2], X[~inds, -1], 'r_')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
                                                                                
        if axis is None:                                                        
            return fig, ax
        
    # Converting to other data formats 
    def add_to_dict(self, node, d):
        d[node.id] = [node.value, node.feature, node.thresh] + \
                     ([-1, -1] if (not node.kids) else [c.id for c in node.kids])
        if node.kids:
            self.add_to_dict(node.kids[0], d)
            self.add_to_dict(node.kids[1], d)
        
    def as_dict(self):
        d = {}
        self.add_to_dict(self.root, d)
        return d
        
    def to_networkx(self):
        d = self.as_dict()
        g = nx.DiGraph({key:val[-2:] for key, val in d.items() if val[-1] != -1})
        nx.set_node_attributes(g, {key:val[0] for key, val in d.items()}, 'value')
        nx.set_node_attributes(g, {key:val[1] for key, val in d.items()}, 'feature')
        nx.set_node_attributes(g, {key:val[2] for key, val in d.items()}, 'threshold')
        return g

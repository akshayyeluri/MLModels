import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from MLModels import utils as u
from MLModels.linearModels import LinearClassifier
from MLModels.cluster import kmeans

def rbf_transform(X, reps, gamma=1):
    '''
    The nonlinear transform associated with radial basis functions.
    Takes a matrix X and a matrix of representatives reps,
    computes the transform matrix M, where 
    M_ij = exp(-gamma * ||x_i - reps_j||^2)
    '''
    X = np.atleast_2d(X)
    reps = np.atleast_2d(reps)
    
    (N,d), (k,d2) = X.shape, reps.shape
    if d != d2:
        raise ValueError('Dimension mismatch for X/reps')
        
    sq_norms = (X[:, np.newaxis, :] - reps[np.newaxis, :, :]) ** 2
    print(sq_norms.shape)
    ans = - gamma * np.sum(sq_norms, axis=2)      
    return np.exp(ans)
        
        
# Radial basis functions, these are a subclass of linear models
# because they can be interpreted as a linear model applied to a
# nonlinear transform of inputs
class RBF(LinearClassifier):
    '''
    Attributes:
        _size -- Number of weights (including w_0, aka bias).
                 Here, _size is k + 1, where k is number of
                 representatives (number of gaussians)
        _weights -- weight vector including w_0, aka bias
        X -- representatives / centers for gaussians
        input_d -- dimensionality of inputs
        gamma -- precision of gaussians
    '''
    def __init__(self, input_d, k=5, gamma=1): 
        super(RBF, self).__init__(k) # sets _size and _weights
        self.gamma = gamma
        self._input_d = input_d
        self.rep = np.zeros((k, input_d))
        
    def signal(self, x):
        sigs = rbf_transform(x, self.rep)        
        return np.dot(self.check_input_dim(sigs), self._weights)
                                                            
    def fit(self, X, Y, **conditions):
        if (X.shape[1] != self._input_d):
            raise ValueError("Dimension mismatch with input X!")
        self.rep = kmeans(X, self._size, **conditions)[0]
        sigs = rbf_transform(X, self.rep)
        return super(RBF, self).fit(sigs, Y, **conditions)
                                                                           
    def plot_reps(self, axis=None, c='k', marker='.', alpha=0.3, s=20):
        if self._input_d != 2:
            raise ValueError('Model not of dimension 2, cannot plot!')
        ax = axis if axis else plt.subplot(111)

        ax.scatter(self.rep[:, -2], self.rep[:, -1],\
                   marker=marker, color=c,\
                   alpha=alpha, s=s, label='Centers for radial basis funcs')

        if axis is None:
            return ax

    def boundary2D_plot(self, color='g', label='Hypothesis', axis=None,\
                        axis_res=200, x1Range=None, x2Range=None):
        if self._input_d != 2:
            raise ValueError('Model not of dimension 2, cannot plot!')
        ax = axis if axis else plt.subplot(111)

        rep = self.reps
        x1Range = x1Range if x1Range else (rep[:, -2].min(), rep[:, -2].max())
        x2Range = x2Range if x2Range else (rep[:, -1].min(), rep[:, -1].max())
        u.plotBoundary(self, x1Range, x2Range, axis=ax, color=color,\
                       axis_res=axis_res, label=label)
        if axis is None:
            return ax


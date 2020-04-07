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
        gamma -- precision of gaussians
    '''
    def __init__(self, k, gamma=1): 
        super(RBF, self).__init__(k) # sets _size and _weights
        self.gamma = gamma
        self.rep = None
                                                            

    def fit(self, X, Y, **conditions):
        self.rep = kmeans(X, self._size, **conditions)[0]
        self.transform = lambda x: rbf_transform(x, self.rep, self.gamma)
        return super(RBF, self).fit(X, Y, **conditions)
                                                                           

    def plot_reps(self, axis=None, c='k', marker='o', alpha=1.0, s=50):
        if (self.rep is None) or self.rep.shape[1] != 2:
            raise ValueError('Model not of dimension 2, cannot plot!')
        ax = axis if axis else plt.subplot(111)

        ax.scatter(self.rep[:, -2], self.rep[:, -1],\
                   marker=marker, color=c,\
                   alpha=alpha, s=s, label='Centers for radial basis funcs')

        if axis is None:
            return ax


    def boundary2D_plot(self, color='g', label='Hypothesis', axis=None,\
                        axis_res=200, x1Range=None, x2Range=None, fontsize=15):
        rep = self.rep
        x1Range = x1Range if x1Range else (rep[:, -2].min(), rep[:, -2].max())
        x2Range = x2Range if x2Range else (rep[:, -1].min(), rep[:, -1].max())
        super(RBF, self).boundary2D_plot(color=color, label=label, axis=axis,\
                                         axis_res=axis_res, x1Range=x1Range,\
                                         x2Range=x2Range, fontsize=fontsize)

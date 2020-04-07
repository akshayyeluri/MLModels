import numpy as np
import matplotlib.pyplot as plt

from MLModels import utils as u
from MLModels.linearModels import LinearClassifier
from MLModels.radialBasis import rbf_transform

# Only for SVMs
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
solvers.options['abstol'] = 1e-10
solvers.options['reltol'] = 1e-10
solvers.options['feastol'] = 1e-10


class SVM(LinearClassifier):

    def __init__(self, d, kernel=None, transform=None, C=np.inf, Q=2, gamma=None):
        super(SVM, self).__init__(d, transform=transform)
        if kernel is None:
            kernel = 'linear'
        if not kernel in ['linear', 'poly', 'rbf']:
            raise TypeError('Improper kernel')
        self.kernel = kernel

        self.C = C # penalty for slack variables in soft margin

        # Set all undefined attributes to None
        self._weights = None # Only for linear kernel
        self.A = None # Alphas from dual
        self.X = None # Support vectors
        self.Y = None # Support vector labels
        self.nSupport = None
        self.bias = None
        self.Q = None # Degree of polynomial for poly kernel
        self.gamma = None # Precision of gaussian for rbf kernel

        if self.kernel == 'poly':
            self.Q = Q
        if self.kernel == 'rbf':
            self.gamma = gamma


    def getInnerProds(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, np.transpose(X2))
        elif self.kernel == 'poly':
            return (1 + np.dot(X1, np.transpose(X2))) ** self.Q
        elif self.kernel == 'rbf':
            return rbf_transform(X1, X2, gamma=self.gamma)

    def signal(self, x):
        '''
        Get the signal (sum over support vectors i of
        a_i * y_i * <x_i, x>
        '''
        x = self.check_input_dim(x, keep_b_col=False)
        sigs = self.getInnerProds(x, self.X)
        return np.dot(sigs, self.A * self.Y) + self.bias


    def fit(self, X, Y, **conditions):
        '''One step learning using Quadratic Programming'''
        X = X if not self.transform else self.transform(X)
        X = self.check_input_dim(X, keep_b_col=False)
        N = X.shape[0]

        P = matrix(np.outer(Y, Y) * self.getInnerProds(X, X))
        q = matrix(-1 * np.ones(N))
        G = -1 * np.eye(N)
        h = np.zeros(N)
        # Soft margin
        if self.C != np.inf:
            G = np.vstack((G, np.eye(N)))
            h = np.hstack((h, np.ones(N) * self.C))
        G, h = matrix(G), matrix(h)
        A = matrix(Y.astype(float), (1, N))
        b = matrix(0.0)

        # Find w
        sol = solvers.qp(P, q, G, h, A, b)
        A = np.array(sol['x']).squeeze()

        # Fix support vector alphas (make most 0, make many C)
        cutoff = conditions.get('cutoff', 1e-4)
        A[A < cutoff] = 0
        if self.C != np.inf:
            A[A > self.C - cutoff] = self.C

        inds = np.where(A > 0)[0]
        # Save many things
        self.A = A[inds].squeeze()
        self.nSupport = inds.shape[0]
        self.X = X[inds]
        self.Y = Y[inds].squeeze()

        # Solve for bias b
        inds = np.where(self.A < self.C)[0]
        if inds.shape[0] == 0:
            self.bias = 0
        else:
            ind = inds[0]
            sigs = self.getInnerProds(self.X[ind], self.X)
            self.bias = self.Y[ind] - np.dot(sigs, self.A * self.Y)

        # If linear kernel (so the model has actual weights) save the w
        if self.kernel == 'linear':
            w = np.array(np.dot(self.A * self.Y, self.X))
            self._weights = np.array([self.bias] + list(w))
        return 1, np.array([self.findE_in(X, Y)])


    def plot_support_vecs(self, axis=None, margin_svec_col='k', marker='o',\
                          nonmargin_svec_col='y', alpha=1.0, s=50):
        if self._size != 3:
            raise ValueError('Model not of dimension 2, cannot plot!')
        ax = axis if axis else plt.subplot(111)

        ax.scatter(self.X[:, -2], self.X[:, -1],\
                   marker=marker, color=margin_svec_col,\
                   alpha=alpha, s=s, label='Support Vectors')

        if self.C != np.inf:
            inds = np.where(self.A == self.C)[0]
            ax.scatter(self.X[inds, -2], self.X[inds, -1],\
                       marker=marker, color=nonmargin_svec_col,\
                       alpha=alpha, s=s, label='Non-Margin Support Vectors')
        if axis is None:
            return ax


    def boundary2D_plot(self, color='g', label='Hypothesis', axis=None,\
                        axis_res=200, x1Range=None, x2Range=None, fontsize=15):
        if self._size != 3:
            raise ValueError('Model not of dimension 2, cannot plot!')
        ax = axis if axis else plt.subplot(111)

        if self._weights is not None:
            u.plotLine(*self.boundary2D(), axis=ax, color=color, label=label)
        else:
            rep = self.X
            x1Range = x1Range if x1Range else (rep[:, -2].min(), rep[:, -2].max())
            x2Range = x2Range if x2Range else (rep[:, -1].min(), rep[:, -1].max())
            u.plotBoundary(self, x1Range, x2Range, axis=ax, color=color,\
                           axis_res=axis_res, label=label, fontsize=fontsize)
        if axis is None:
            return ax


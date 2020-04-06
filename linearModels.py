import numpy as np
import matplotlib.pyplot as plt

from MLModels import utils as u

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
solvers.options['abstol'] = 1e-10
solvers.options['reltol'] = 1e-10
solvers.options['feastol'] = 1e-10



class LinearModel():
    '''
    BaseClass for all LinearModels, uses linear regression as the learning
    algorithm and mean squared error for the loss function.
    '''
    def __init__(self, d, weights=None):
        '''
        Initialize the weights and number of parameters for a linear model
        '''
        self._size = d + 1 # Extra 1 for bias
        if weights is None:
            self._weights = np.zeros(self._size).astype(float)
        elif len(weights) == d: # Assume they forgot to provide a bias
            self._weights = np.array([0.0] + list(weights))
        else:
            if len(weights) != self._size:
                raise ValueError("Initial weight vector of wrong length")
            self._weights = np.array(weights).astype(float)

    def check_input_dim(self, X, keep_b_col=True):
        '''
        Handles the conversion between an explicit bias term in linear
        models, and incorporating the bias by having an extra w_0 weight,
        and concatentating 1 to the front of inputs x. Also checks
        the dimensionality of X against the models size (self._size)

        Args:
            keep_b_col: Boolean flag, set to true to interpret X
                        as having an extra 1 at the front / w_0 weight
                        Set to false to get back X without ones column
        '''
        if (not keep_b_col) and X.shape[1] == self._size and np.all(X[:, 0] == 1):
            X = X[:, 1:]
        elif (keep_b_col) and X.shape[1] == self._size - 1:
            X = np.hstack((np.ones(shape=(X.shape[0], 1)), X))
        if X.shape[1] != (self._size if keep_b_col else (self._size - 1)):
            raise ValueError('Bad data dimensions!')
        return X

    def signal(self, x):
        '''
        Calculate the signal of a linear model. Vectorized, can calculate
        a vector of signals for an input matrix.
        '''
        return np.dot(x, self._weights)

    def __call__(self, x):
        '''Calculate the hypothesis value for x'''
        return self.signal(self.check_input_dim(x))

    def err(self, x, y):
        '''
        Find the pointwise loss function, given point and correct output.
        Vectorized, can calculate a vector of N errors for N inputs and outputs.
        '''
        return (self(x) - y) ** 2


    def findE_in(self, X, Y):
        '''
        Find the total loss (the in sample error),
        given training data with correct outputs
        '''
        return np.mean(self.err(X, Y))


    # Assuming that X has column of ones to account for bias
    def fit(self, X, Y, **conditions):
        '''One step learning using linear regression'''
        X = self.check_input_dim(X)
        X_dagger = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
        self._weights = np.dot(X_dagger, Y)
        return 0, np.array([]) # Return nothing to be consistent


    def boundary2D(self):
        if self._size > 3:
            raise ValueError('Not a 2D model!')
        elif self._weights is None:
            raise ValueError('No weights in this model!')
        return list(self._weights[1:]) + [self._weights[0]]

    def isDone(self, it, w_old, E_in, maxIters=None, wDiffBound=None,\
                                                     errBound=None):
        '''Check many termination conditions to decide if learning is done.'''
        if maxIters is not None and it >= maxIters:
            return True

        wDiff = np.linalg.norm(self._weights - w_old)
        if wDiffBound is not None and wDiff <= wDiffBound:
            return True

        if errBound is not None and E_in <= errBound:
            return True

        return False


class Perceptron(LinearModel):
    '''
    The Perceptron Model (a binary classifier), uses PLA as the learning
    algorithm, and binary error as the loss function.
    '''
    def __call__(self, x):
        '''Calculate the hypothesis value for x (sign of the signal)'''
        s = self.signal(self.check_input_dim(x))
        return np.sign(s)


    def err(self, x, y):
        '''Binary error'''
        return self(x) != y


    def findE_in(self, X, Y):
        '''
        Like the general findE_in for linear models, but keep track of
        which points in the input are misclassified.
        '''
        errs = self.err(X, Y)
        self._badInds = np.where(errs)[0]
        return np.mean(errs)


    def fit(self, X, Y, **conditions):
        '''
        Given training data, will learn from it. Will iteratively use the
        PLA algorithm to update the weights. Set trackE_in to true to calculate
        E_in after every iteration. Will continue updating till one of the
        conditions is met (conditions can be passed in as keyword arguments,
        and include a max number of iterations, a bound on E_in, etc.)
        '''
        X = self.check_input_dim(X)

        # Get termination conditions
        maxIters = conditions.get('maxIters', None)
        errBound = conditions.get('errBound', 0.0)
        wDiffBound = conditions.get('wDiffBound', None)

        # Define variables pertaining to termination
        it = 0
        w_old = self._weights.copy()
        E_ins = [self.findE_in(X, Y)]

        # Initial check for termination
        if errBound is not None and E_ins[-1] <= errBound:
            return it, np.array(E_ins)

        while True:
            w_old = self._weights.copy()

            ind = np.random.choice(self._badInds)
            self._weights += Y[ind] * X[ind]

            # Update termination relevant variables
            it += 1
            E_ins.append(self.findE_in(X, Y))

            # Check if to terminate
            if self.isDone(it, w_old, E_ins[-1],\
                           maxIters, wDiffBound, errBound):
                return it, np.array(E_ins)



class LogisticRegression(LinearModel):
    '''
    The Logistic Regression Model (outputs 0 to 1), uses stochastic or batch
    gradient descent as the learning algorithm, and cross entropy error as
    the loss function.
    '''
    def theta(self, x):
        return np.exp(x) / (1 + np.exp(x))


    def __call__(self, x):
        '''Calculate the hypothesis value for x (theta of the signal)'''
        s = self.signal(self.check_input_dim(x))
        return self.theta(s)


    def err(self, x, y):
        '''Cross entropy error'''
        return np.log(1 + np.exp(-y * self.signal(x)))


    def fit(self, X, Y, eta=0.1, useBatch=False, **conditions):
        '''
        Given training data, will learn from it. Will use either stochastic
        or batch gradient descent to update the weights.
        Set trackE_in to true to calculate
        E_in after every epoch. Will continue updating till one of the
        conditions is met (conditions can be passed in as keyword arguments,
        and include a max number of epochs, a bound on the change in weights,
        etc.)
        '''

        X = self.check_input_dim(X)

        # Get termination conditions
        maxIters = conditions.get('maxIters', None)
        errBound = conditions.get('errBound', None)
        wDiffBound = conditions.get('wDiffBound', 0.01)

        # Define variables pertaining to termination
        it = 0
        w_old = self._weights.copy()
        E_ins = [self.findE_in(X, Y)]

        inds = np.arange(X.shape[0])

        # Initial check for termination
        if errBound is not None and E_ins[-1] <= errBound:
            return it, np.array(E_ins)

        while True:
            w_old = self._weights.copy()

            if useBatch:
                s = -Y * self.signal(X)
                grad = np.dot((-Y * self.theta(s)), X)
                self._weights -= eta * grad
            else:
                np.random.shuffle(inds)
                for i in inds:
                    x, y = X[i], Y[i]
                    s = - y * self.signal(x)
                    grad = -y * self.theta(s) * x
                    self._weights -= eta * grad

            # Update termination relevant variables
            it += 1
            E_ins.append(self.findE_in(X, Y))

            # Check if to terminate
            if self.isDone(it, w_old, E_ins[-1],\
                           maxIters, wDiffBound, errBound):
                return it, np.array(E_ins)


class SVM(LinearModel):

    def __init__(self, d, kernel=None, C=np.inf, Q=2, gamma=None):
        self._size = d + 1 # Number of dimensions (including bias dim)
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
            return rb.features(X1, X2, gamma=self.gamma)

    def signal(self, x):
        '''
        Get the signal (sum over support vectors i of
        a_i * y_i * <x_i, x>
        '''
        sigs = self.getInnerProds(x, self.X)
        return np.dot(sigs, self.A * self.Y) + self.bias

    def __call__(self, x):
        '''Calculate the hypothesis value for x (sign of the signal)'''
        return np.sign(self.signal(self.check_input_dim(x, keep_b_col=False)))

    def err(self, x, y):
        '''Binary error'''
        return self(x) != y

    def fit(self, X, Y, **conditions):
        '''One step learning using Quadratic Programming'''
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
        self.A = A[inds]
        self.nSupport = inds.shape[0]
        self.X = X[inds]
        self.Y = Y[inds]

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


    def plot_support_vecs(self, axis=None, margin_svec_col='k', marker='.',\
                          nonmargin_svec_col='y', alpha=0.3, s=20):
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
                        axis_res=200, atol=1e-8, x1Range=None, x2Range=None):
        if self._size != 3:
            raise ValueError('Model not of dimension 2, cannot plot!')
        ax = axis if axis else plt.subplot(111)

        if self._weights is not None:
            u.plotLine(*self.boundary2D(), axis=ax, color=color, label=label)
        else:
            x1Range = x1Range if x1Range else (self.X[:, -2].min(), self.X[:, -2].max())
            x2Range = x2Range if x2Range else (self.X[:, -1].min(), self.X[:, -1].max())
            x1 = np.linspace(*x1Range, axis_res)
            x2 = np.linspace(*x2Range, axis_res)
            X1, X2 = np.meshgrid(x1, x2)
            X_n = np.hstack((X1.flatten()[:, np.newaxis],\
                             X2.flatten()[:, np.newaxis]))
            sigs = np.array(self.signal(X_n)).reshape(X1.shape)
            ax.contour(X1, X2, sigs, levels=[0], colors=color)
        if axis is None:
            return ax


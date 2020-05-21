import numpy as np
import matplotlib.pyplot as plt

from MLModels import utils as u

class LinearModel():
    '''
    BaseClass for all LinearModels, uses linear regression as the learning
    algorithm and mean squared error for the loss function.
    '''
    def __init__(self, d, weights=None, transform=None):
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

        if not (transform is None or callable(transform)):
            raise TypeError('Improper (non-function) argument for transform')
        self.transform = transform

    def check_input_dim(self, X, keep_b_col=True, dim=None):
        '''
        Handles the conversion between an explicit bias term in linear
        models, and incorporating the bias by having an extra w_0 weight,
        and concatentating 1 to the front of inputs x. Also checks
        the dimensionality of X against the models size (self._size)

        Args:
            keep_b_col: Boolean flag, set to true to interpret X
                        as having an extra 1 at the front / w_0 weight
                        Set to false to get back X without ones column
            dim: the expected length of input vectors
                 (including the 1 at the front for the bias term).
                 leave as None to use the model's _size attribute.
        '''
        dim = dim if dim else self._size
        X = np.atleast_2d(X)
        if (not keep_b_col) and X.shape[1] == dim and np.all(X[:, 0] == 1):
            X = X[:, 1:]
        elif (keep_b_col) and X.shape[1] == dim - 1:
            X = np.hstack((np.ones(shape=(X.shape[0], 1)), X))
        if X.shape[1] != (dim if keep_b_col else (dim - 1)):
            raise ValueError('Bad data dimensions!')
        return X.squeeze()

    def signal(self, x):
        '''
        Calculate the signal of a linear model. Vectorized, can calculate
        a vector of signals for an input matrix.
        '''
        return np.dot(self.check_input_dim(x), self._weights)

    def __call__(self, x, do_transform=True):
        '''Calculate the hypothesis value for x'''
        x = self.transform(x) if (self.transform and do_transform) else x
        return self.signal(x)

    def err(self, x, y, do_transform=True):
        '''
        Find the pointwise loss function, given point and correct output.
        Vectorized, can calculate a vector of N errors for N inputs and outputs.
        '''
        return (self(x, do_transform) - y) ** 2


    def findE_in(self, X, Y, do_transform=True):
        '''
        Find the total loss (the in sample error),
        given training data with correct outputs
        '''
        return np.mean(self.err(X, Y, do_transform))


    def fit(self, X, Y, **conditions):
        '''One step learning using linear regression'''
        X = X if not self.transform else self.transform(X)
        X = self.check_input_dim(X)
        self._weights = np.dot(np.linalg.pinv(X), Y)
        # Return to be consistent
        return 1, np.array([self.findE_in(X, Y, do_transform=False)])


    def boundary2D(self):
        if self._size > 3:
            raise ValueError('Not a 2D model!')
        elif self._weights is None:
            raise ValueError('No weights in this model!')
        return list(self._weights[1:]) + [self._weights[0]]


    def boundary2D_plot(self, color='g', label='Hypothesis', axis=None,\
                        axis_res=200, x1Range=None, x2Range=None, fontsize=15):
        ax = axis if axis else plt.subplot(111)

        if self._size == 3 and (not self.transform):
            u.plotLine(*self.boundary2D(), axis=ax, color=color, label=label)
        elif self.transform:
            x1Range = x1Range if x1Range else [-1, 1]
            x2Range = x2Range if x2Range else [-1, 1]
            u.plotBoundary(self, x1Range, x2Range, axis=ax, fontsize=fontsize,\
                           transform=self.transform, color=color, label=label)
        else:
            raise ValueError('Model not of dimension 2, cannot plot!')

        if axis is None:
            return ax


    def isDone(self, it, w_old, E_in, maxIters=None, wDiffBound=None,\
                    errBound=None, errDiffBound=None, E_in_old=None):
        '''Check many termination conditions to decide if learning is done.'''
        if maxIters is not None and it >= maxIters:
            return True

        wDiff = np.linalg.norm(self._weights - w_old)
        if wDiffBound is not None and wDiff <= wDiffBound:
            return True

        if errBound is not None and E_in <= errBound:
            return True

        if errDiffBound is not None and E_in_old is not None and \
           E_in_old - E_in <= errDiffBound:
            return True

        return False

    def __repr__(self):
        return type(self).__name__


class LinearClassifier(LinearModel):
    def __call__(self, x, do_transform=True):
        '''Calculate a hypothesis by taking the sign of the signal'''
        return np.sign(super(LinearClassifier, self).__call__(x, do_transform))

    def err(self, x, y, do_transform=True):
        '''Use binary error instead of squared error'''
        return self(x, do_transform) != y


class Perceptron(LinearClassifier):
    '''
    The Perceptron Model (a binary classifier), uses PLA as the learning
    algorithm
    '''

    def findE_in(self, X, Y, do_transform=True):
        '''
        Like the general findE_in for linear models, but keep track of
        which points in the input are misclassified.
        '''
        errs = self.err(X, Y, do_transform)
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
        X = X if not self.transform else self.transform(X)
        X = self.check_input_dim(X)

        # Get termination conditions
        maxIters = conditions.get('maxIters', None)
        errBound = conditions.get('errBound', 0.0)
        wDiffBound = conditions.get('wDiffBound', None)

        # Define variables pertaining to termination
        it = 0
        w_old = self._weights.copy()
        E_ins = [self.findE_in(X, Y, do_transform=False)]

        # Initial check for termination
        if errBound is not None and E_ins[-1] <= errBound:
            return it, np.array(E_ins)

        while True:
            w_old = self._weights.copy()

            ind = np.random.choice(self._badInds)
            self._weights += Y[ind] * X[ind]

            # Update termination relevant variables
            it += 1
            E_ins.append(self.findE_in(X, Y, do_transform=False))

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
        return 1 / (1 + np.exp(-x))


    def __call__(self, x, do_transform=True):
        '''Calculate the hypothesis value for x (theta of the signal)'''
        x = self.transform(x) if (self.transform and do_transform) else x
        s = self.signal(x)
        return self.theta(s)


    def err(self, x, y, do_transform=True):
        '''Cross entropy error'''
        x = self.transform(x) if (self.transform and do_transform) else x
        return np.log(1 + np.exp(-y * self.signal(x)))


    def fit(self, X, Y, eta=0.1, useBatch=False, useIRLS=True, **conditions):
        '''
        Given training data, will learn from it. Will use
        either stochastic / batch gradient descent to update the weights.
        (OR IRLS if useIRLS is True, note that this will convert Y to zero_one output)
        Set trackE_in to true to calculate
        E_in after every epoch. Will continue updating till one of the
        conditions is met (conditions can be passed in as keyword arguments,
        and include a max number of epochs, a bound on the change in weights,
        etc.)
        '''

        X = X if not self.transform else self.transform(X)
        X = self.check_input_dim(X)

        # Get termination conditions
        maxIters = conditions.get('maxIters', None)
        errBound = conditions.get('errBound', None)
        errDiffBound = conditions.get('errDiffBound', 1e-5)
        wDiffBound = conditions.get('wDiffBound', 0.01)

        # Define variables pertaining to termination
        it = 0
        w_old = self._weights.copy()
        E_ins = [self.findE_in(X, Y, do_transform=False)]

        if useIRLS:
            Y = u.conv_bin_labels(Y, to_zero_one=True)
        inds = np.arange(X.shape[0])

        # Initial check for termination
        if errBound is not None and E_ins[-1] <= errBound:
            return it, np.array(E_ins)

        while True:
            w_old = self._weights.copy()
            E_in_old = E_ins[-1]

            if useIRLS:
                P = self.theta(self.signal(X))
                W = np.diag((1-P) * P)
                to_invert = X.T @ W @ X
                # Break before we hit numerical issues
                if (np.linalg.matrix_rank(to_invert) < self._size):
                    print("Singular matrix found during iteration!")
                    return it, np.array(E_ins)
                self._weights += np.linalg.inv(to_invert) @ X.T @ (Y - P)
            elif useBatch:
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
            E_ins.append(self.findE_in(X, Y, do_transform=False))

            # Check if to terminate
            if self.isDone(it, w_old, E_ins[-1], maxIters, wDiffBound,
                           errBound, errDiffBound, E_in_old):
                return it, np.array(E_ins)

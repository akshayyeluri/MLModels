'''
Utilities for learning models
'''

import numpy as np
import matplotlib.pyplot as plt

######################################################
# Data manipulation / Error metrics
######################################################

def to_categorical(x):
    return (np.argmax(x, 1) if x.ndim == 2 else x).astype(int)

def error_rate(predictions, labels):
    """Return the error rate (fraction of samples misclassified)"""
    correct = np.sum(to_categorical(predictions) == to_categorical(labels))
    total = predictions.shape[0]
    return (1  - float(correct) / float(total))

def confusions(predictions, labels):
    """Return the confusions matrix"""
    cat_preds = to_categorical(predictions)
    confusions = np.zeros([max(cat_preds) + 1] * 2, np.float32)
    bundled = zip(to_categorical(predictions), to_categorical(labels))
    for predicted, actual in bundled:
        confusions[predicted, actual] += 1
    return confusions

def onehot(data, n_class = None):
    '''Process labels, make one-hot, fix 1-indexing, etc.'''
    n_class = len(np.unique(data)) if n_class is None else n_class
    data = np.round(((data-data.min()) / (np.ptp(data)) * (n_class-1)).squeeze())
    # One hot the data
    data = (np.arange(n_class) == data[:, None]).astype(np.float32)
    return data

def perf_metric(predictions, labels, metric='precision'):
    """
    A generic function to return a performance metric given the
    predictions and labels.
    Supported metrics:
        - accuracy
        - error
        - precision (per class)
        - recall (per class)
    """
    conf = confusions(predictions, labels)
    if metric == 'precision':
        return np.diag(conf) / np.sum(conf, axis=1)
    elif metric == 'recall':
        return np.diag(conf) / np.sum(conf, axis=0)
    elif metric == 'accuracy':
        return np.sum(np.diag(conf)) / np.sum(conf)
    elif metric == 'error':
        return 1 - (np.sum(np.diag(conf)) / np.sum(conf))



######################################################
# Generating data / functions
######################################################

def genPoints(nPoints=1):
    ''' Generate points in interval [-1, 1] x [-1, 1]'''
    return np.random.rand(nPoints, 2) * 2 - 1

def conv_bin_labels(labels, to_zero_one=True):
    if to_zero_one:
        return (labels + 1) // 2
    return labels * 2 - 1

def genF(zero_one=False):
    '''
    Generate a random function f: [-1, 1] x [-1, 1] -> {-1, 1}. Returns both
    this function, and the line it defines in space, which is represented as
    a 3-tuple of the form (a, b, c), for a line ax + by + c = 0. Set zero_one
    to True to have the range of this function be {0, 1} instead
    '''
    # Pick 2 points, find vector between them
    ps = genPoints(2)
    v0 = ps[1] - ps[0]

    # define a function that maps point on one side of this vector to -1,
    # other side to +1, by computing a determinant
    def f(point):
        v1 = point - ps[0]
        return np.sign(np.linalg.det([v0, v1]))
    line = (v0[1], -v0[0], v0[0] * ps[0, 1] - v0[1] * ps[0, 0])
    if not zero_one:
        return f, line
    return (lambda x: (f(x) + 1) // 2), line


def genCircleF(r=None, zero_one=False):
    '''
    Generate a function f: R^2 -> {-1, 1} (or {0,1} if zero_one is True),
    that is a binary classifier,
    where points outside a circle of radius r are mapped to +1, and points
    inside are mapped to -1. Returns this function, and the circle radius.
    '''
    r = r if r else np.random.normal(0.75, 0.1)

    def f(X):
        X = np.atleast_2d(X)
        return np.sign(X[:, 0] ** 2 + X[:, 1] ** 2 - r ** 2).squeeze()
    if not zero_one:
        return f, r
    return (lambda x: (f(x) + 1) // 2), r


def genData(f, n, appendOnes=False):
    '''
    Given a target function f, generate training data, where 1 is optionally
    appended to the points (to make inputs 3-d), and outputs are what f maps each
    point to.
    '''
    points = genPoints(n)
    outputs = np.array([f(point) for point in points])
    if not appendOnes:
        return points, outputs
    return np.hstack((np.ones(shape=(n, 1)), points)), outputs

def genTransform(k=3):
    '''
    Generates a non-linear transform function, where the number
    of produced features depends on input k
    '''
    def transform(X):
        X = np.atleast_2d(X)
        X_out = np.empty((X.shape[0], k))
        try:
            X_out[:, 0] = X[:, -2]
            X_out[:, 1] = X[:, -1]
            X_out[:, 2] = X[:, -2] ** 2
            X_out[:, 3] = X[:, -1] ** 2
            X_out[:, 4] = X[:,-2] * X[:, -1]
            X_out[:, 5] = np.abs(X[:,-2] - X[:, -1])
            X_out[:, 6] = np.abs(X[:,-2] + X[:, -1])
        except IndexError as e:
            pass
        return X_out
    return transform

######################################################
# Plotting utilities
######################################################

def plotLine(a, b, c, label='', axis=None, nPoints=3, color='k'):
    '''
    Given the a, b, c parameters of a line of the form ax + by + c = 0,
    will plot this line on axis, or will create a [-1, 1] x [-1, 1] axis and
    plot on that.
    '''
    x = np.linspace(-1, 1, nPoints)
    if b == 0:
        if a == 0:
            return
        y = np.linspace(-1, 1, nPoints)
        x = np.full(y.shape, -c / a)
    else:
        y = (-a / b) * x - (c / b)

    ax = axis if axis else plt.subplot(111)
    if axis is None:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    line = ax.plot(x, y, label=label, color=color)
    if axis is None:
        return ax


def plotBoundary(model, x1Range, x2Range, axis=None, transform=None,\
                 color='g', label='hypothesis', axis_res=200, fontsize=15):
    '''
    Given a model and an x1 and x2 range, this will plot the 2d
    decision boundary of the model. For example, use this to find
    the 2d decision boundary of an SVM with a kernel function
    '''
    ax = axis if axis else plt.subplot(111)

    x1 = np.linspace(*x1Range, axis_res)
    x2 = np.linspace(*x2Range, axis_res)
    X1, X2 = np.meshgrid(x1, x2)
    X_n = np.hstack((X1.flatten()[:, np.newaxis],\
                     X2.flatten()[:, np.newaxis]))
    X_n = X_n if not transform else transform(X_n)
    sigs = np.array(model.signal(X_n)).reshape(X1.shape)
    CS = ax.contour(X1, X2, sigs, levels=[0], colors=color)
    ax.clabel(CS, [0], fmt={0.:label}, inline=True, fontsize=fontsize)

    if axis is None:
        return ax


def plotE_ins(E_ins, axis=None):
    '''Plots E_in progression over iterations'''
    ax = axis if axis else plt.subplot(111)

    ax.plot(np.arange(len(E_ins)), E_ins)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('E_in')

    if axis is None:
        return ax

def quickScatter(X, Y, axis=None, alpha=1.0):
    '''
    For two dimensional input data and corresponding outputs,
    this will plot all the points in the data, useful for quick visualization
    '''
    if X.shape[1] not in (2, 3):
        raise ValueError('plotting requires 2D input data')

    ax = axis if axis else plt.subplot(111)

    inds = (Y == 1)
    ax.plot(X[inds, -2], X[inds, -1], 'b+', alpha=alpha)
    ax.plot(X[~inds, -2], X[~inds, -1], 'r_', alpha=alpha)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_xlim([X[:, -2].min(), X[:, -2].max()])
    ax.set_xlim([X[:, -1].min(), X[:, -1].max()])

    if axis is None:
        return ax

def plot_confusions(grid, axis = None):
    """ Utility to neatly plot confusions matrix. """
    ax = axis if axis else plt.subplot(111)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.grid(False)
    ax.set_xticks(np.arange(grid.shape[0]))
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.imshow(grid, interpolation='nearest');

    for i, cas in enumerate(grid):
        for j, count in enumerate(cas):
            if count > 0:
                xoff = .07 * len(str(count))
                plt.text(j-xoff, i+.2, int(count), fontsize=9, color='white')

    if axis is None:
        return ax

def rand_hex():
    return '#%02X%02X%02X' % tuple(np.random.randint(256, size=3))

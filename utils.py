'''
Utilities for learning models
'''

import numpy as np
import matplotlib.pyplot as plt

def genPoints(nPoints=1):
    ''' Generate points in interval [-1, 1] x [-1, 1]'''
    return np.random.rand(nPoints, 2) * 2 - 1

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


def genData(f, n, appendOnes=True):
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
    if axis is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    else:
        ax = axis
    line = ax.plot(x, y, label=label, color=color)
    if axis is None:
        return fig, ax


def plotE_ins(E_ins, axis=None):
    '''Plots E_in progression over iterations'''
    if axis is None:
        fig, ax = plt.subplots(1,1)
    else:
        ax = axis
    
    ax.plot(np.arange(len(E_ins)), E_ins)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('E_in')

    if axis is None:
        return fig, ax

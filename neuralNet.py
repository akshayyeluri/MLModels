import numpy as np
import matplotlib.pyplot as plt

import utils as u

class NeuralNet():
    
    def __init__(self, sizes, eta=0.1, theta=np.tanh,\
                 dTheta_ds=lambda x: 1 - (np.tanh(x) ** 2),\
                 E=lambda x,y: (y - x) ** 2,\
                 dE_dTheta=lambda x,y: -2 * (y - x)):
        '''
        Initialize a neural net, with the number of layers and size of 
        each layer, as well as various other specifications. 
        
        Arguments:
            - sizes: array-like, shape (nLevels + 1,). Gives the size of each 
                  layer of the neural net
            - eta: scalar (float) learning rate, defaults to 0.1
            - theta: function, applied to the signal at each level to introduce 
                  nonlinearity into the model
            - dTheta_ds: function, the derivative of the theta function
            - E: the pointwise error/loss function, returns the error of the 
                  final output (x) vs correct output (y)
            - dE_dTheta: the derivative of E, where theta(final signal) = 
                  final output (a.k.a the x)
        '''
        sizes = np.array(sizes)
        if sizes[-1] != 1:
            raise ValueError('Last layer must have size 1')
        if not np.all([callable(func) for func in \
                      (theta, dTheta_ds, E, dE_dTheta)]):
            raise TypeError('Improper function arguments')
            
        self.sizes = sizes
        self.nLevels = sizes.shape[0] - 1

        self.weights = [np.random.randn(sizes[l], sizes[l + 1]).squeeze()\
                        * np.sqrt(1 / sizes[l]) for l in range(self.nLevels)]
        self.eta = eta
        self.theta = theta
        self.dTheta_ds = dTheta_ds
        self.E = E
        self.dE_dTheta = dE_dTheta
        
        
    def calculate(self, x):
        '''
        Will calculate the final output of the neural net given initial x
        (will also save all signals and outputs in the hidden layers), for use
        in endeavors like the backpropagation algorithm.
        '''
        if self.sizes[0] != 1 and x.shape[0] != self.sizes[0]:
            raise ValueError('Input vector should be of length {}.'.format\
                             (len(x), self.sizes[0]))
        
        X, S = [x], []
        for l in range(self.nLevels):
            S.append(np.dot(X[-1], self.weights[l]))
            X.append(self.theta(S[-1]))
        
        self.X = X
        self.S = S
        return self.X[-1]
        
        
    def err(self, x, y):
        '''Find the pointwise loss, given a point and correct output'''
        return self.E(self.calculate(x), y)
       
        
    def findE_in(self, X, Y):
        '''Find the total loss (the in sample error) 
        given training data with correct outputs'''
        return np.mean([self.err(X[i], Y[i])\
                        for i in range(X.shape[0])])
    
    
    def backPropogate(self, x, y):
        '''
        Given a point, and the correct output, will use the backpropagation 
        algorithm to calculate the partial derivatives with respect to each 
        weight, and will update all the weights
        '''
        self.calculate(x)
        deltas = [np.empty_like(self.S[l]) for l in range(self.nLevels)]
        deltas[-1] = self.dE_dTheta(self.X[-1], y) *\
                     self.dTheta_ds(self.S[-1])
        for l in reversed(range(1, self.nLevels)):
            deltas[l - 1] = self.dTheta_ds(self.S[l - 1]) *\
                            np.dot(self.weights[l], deltas[l])
        for l in range(self.nLevels):
            dw = np.outer(self.X[l], deltas[l]).squeeze()
            self.weights[l] -= self.eta * dw


    def isDone(self, it, w_old, E_in, maxIters=None, wDiffBound=None,\
                                                     errBound=None):
        '''Check many termination conditions to decide if learning is done.'''
        if maxIters is not None and it >= maxIters:
            return True
        
        wDiff = np.sum([np.linalg.norm(self.weights[l] - w_old[l])\
                        for l in range(self.nLevels)])
        if wDiffBound is not None and wDiff <= wDiffBound:
            return True
        
        if errBound is not None and E_in <= errBound:
            return True
        
        return False

    
    
    def learn(self, X, Y, trackE_in=False, **conditions):
        '''
        Given training data, will learn from it. Will iteratively use the 
        backpropagation algorithm to update the weights, going through all 
        examples just once in every epoch. Set trackE_in to true to calculate
        E_in after every epoch. Will continue updating till one of the
        conditions is met (conditions can be passed in as keyword arguments,
        and include a max number of iterations, a minimum degree of difference 
        in the weights between epochs, etc.)
        '''
        # Get termination conditions
        maxIters = conditions.get('maxIters', None)                           
        errBound = conditions.get('errBound', None)
        wDiffBound = conditions.get('wDiffBound', 0.01)
        
        # Define variables pertaining to termination
        it = 0
        w_old = self.weights.copy()
        E_ins = []
        if trackE_in:
            E_ins.append(self.findE_in(X, Y))
        else:
            errBound = None

        inds = np.arange(X.shape[0])
        
        # Initial check for termination
        if errBound is not None and E_ins[-1] <= errBound:
            return it, np.array(E_ins)

        while True:
            w_old = [layer.copy() for layer in self.weights]

            np.random.shuffle(inds)
            for i in inds:
                x, y = X[i], Y[i]
                self.backPropogate(x, y)
            
            # Update termination relevant variables
            it += 1
            if trackE_in:
                E_ins.append(self.findE_in(X, Y))
                E_in = E_ins[-1]
            else:
                E_in = None
                
            # Check if to terminate
            if self.isDone(it, w_old, E_in,\
                           maxIters, wDiffBound, errBound):
                return it, np.array(E_ins)
        
        
    def quickPlot(self, X, Y, color='g', label='Net Outputs', axis=None):
        '''
        For 1 parameter inputs only (scalar inputs), will
        graph neural net performance on data set vs correct outputs'''
        if axis is None:
            fig, ax = plt.subplots(1, 1)
        else:
            ax = axis
            
        ax.plot(X, [self.calculate(i) for i in X],\
                color=color, label=label)
        ax.plot(X, Y, color='k', label='Real Outputs')
        ax.set_xlabel('input')
        ax.set_ylabel('output')
        ax.legend()
        
        if axis is None:
            return fig, ax

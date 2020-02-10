import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from MLModels import utils as u

losses = {
    # Categorical cross entropy,
    'categorical': {
        'f' : lambda t,y: st.entropy(y, t) + st.entropy(y), 
        'df': lambda t,y: -y / t 
    },
    # Binary cross entropy,
    'log': {
        'f' : lambda t,y: st.entropy([y,1-y], [t, 1-t]) + st.entropy([y,1-y]), 
        'df': lambda t,y: (1-y)/(1-t) - y/t # deriv of binary cross entropy
    },
    'square': {
        'f' : lambda x,y: (y - x) ** 2,
        'df': lambda x,y: -2 * (y - x)
    }
}


activations = {
    'softmax': {
        'f' : lambda s :  np.exp(s) / np.exp(s).sum(), # softmax activation function
        'df': lambda s :  np.diag(np.exp(s)/np.exp(s).sum()) - \
                          np.outer(np.exp(s), np.exp(s)) / (np.exp(s).sum() ** 2)
    },
    'sigmoid': {
        'f' : lambda s : 1 / (1 + np.exp(-s)), # sigmoid activation function
        #'df': lambda s: np.exp(-s) / (1 + np.exp(-s)) ** 2,  # Deriv of sigmoid
        'df': lambda s: 1 / (1 + np.exp(s)) * 1 / (1 + np.exp(-s)),  
    },
    'tanh': {
        'f' : np.tanh,
        'df': lambda s: 1 - (np.tanh(s) ** 2),
    },
    'relu': {
        'f' : lambda s: np.maximum(0, s),
        'df': lambda s: (s >= 0).astype(int)
    },
    'None': {
        'f' : lambda s: s,
        'df': lambda s: 1,
    }
}


class NeuralNet():
    
    def __init__(self, sizes, eta=0.1, w_init=None,\
                 loss='square', nonLin='tanh'):
        '''
        Initialize a neural net, with the number of layers and size of 
        each layer, as well as various other specifications. 
        
        Arguments:
            - sizes: array-like, shape (nLevels + 1,). Gives the size of each 
                  layer of the neural net
            - eta: scalar (float) learning rate, defaults to 0.1
            - loss: string giving type of loss, available losses are in 
                  the losses dictionary of this module
            - nonLin: string or list of strings giving the activation 
                  functions at each layer. If single string, will use same
                  activation function at each layer. If list, must be of
                  length len(sizes) - 1 (one activation for each layer). To
                  forego an activation function at a specific layer, use
                  'None'. Available activations are in activations dict of
                  this module
        '''
        self._sizes = np.array(sizes)
        self._nLevels = self._sizes.shape[0] - 1

        self._eta = eta

        # He et. al suggested initializer by default
        if w_init is None:
            initializer = \
             lambda l,l_p1: np.random.randn(l, l_p1).squeeze() * np.sqrt(2/l)
        if isinstance(w_init, (int, float)):
            initializer = \
             lambda l,l_p1: w_init * np.random.randn(l, l_p1).squeeze() * np.sqrt(2/l)
        elif callable(w_init):
            initializer = w_init 
        self._weights = [initializer(sizes[l], sizes[l+1]) for l in range(self._nLevels)]

        # Handle retrieving activations / derivatives of activations
        if isinstance(nonLin, str):
            nonLin = [nonLin] * self._nLevels
        if len(nonLin) != self._nLevels:
            raise ValueError('Number of activation functions and layers mismatch!')
        if not np.all([(t in activations) for t in nonLin]):
            raise ValueError('Unimplemented or unknown nonlinearity!')
        self._thetas = [activations[t]['f'] for t in nonLin]
        self._dThetas = [activations[t]['df'] for t in nonLin]
        self._L = losses[loss]['f']
        self._dL = losses[loss]['df']
        
    @property
    def nWeights(self):
        if not hasattr(self, '_nWeights'):
            self._nWeights = sum([np.prod(l.shape) for l in self._weights])
        return self._nWeights

        
    def calculate(self, x, train=False):
        '''
        Will calculate the final output of the neural net given initial x
        (will also save all signals and outputs in the hidden layers), for use
        in endeavors like the backpropagation algorithm.
        '''
        if self._sizes[0] != 1 and x.shape[0] != self._sizes[0]:
            raise ValueError('Input vector should be of length {}.'.format\
                             (len(x), self._sizes[0]))

        currX, currS = x, 0
        X, S = [x], []
        for l in range(self._nLevels):
            currS = np.dot(currX, self._weights[l])
            currX = self._thetas[l](currS)
            if train:
                S.append(currS)
                X.append(currX)
        
        if train:
            self._X = X
            self._S = S
        return currX
        
        
    def err(self, x, y):
        '''Find the pointwise loss, given a point and correct output'''
        return self._L(self.calculate(x), y)

    def __call__(self, X):
        if X.ndim == 1 and self._sizes[0] > 1:
            return self.calculate(X)
        return np.array([self.calculate(X[i])\
                        for i in range(X.shape[0])])

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
        self.calculate(x, train=True)
        deltas = [np.empty_like(self._S[l]) for l in range(self._nLevels)]

        # We allow for an activation function using all signals coming into
        # a layer (e.g. softmax) only in the last layer. Thus, for this
        # we need a jacobian J where J_{ij} = dOutput_i / dSignal_j 
        nonLin_grad = np.atleast_1d(self._dThetas[-1](self._S[-1]))
        jacobian = nonLin_grad if nonLin_grad.ndim == 2 else np.diag(nonLin_grad)
        deltas[-1] = np.dot(self._dL(self._X[-1], y), jacobian).squeeze()
                     
        for l in reversed(range(1, self._nLevels)):
            deltas[l - 1] = self._dThetas[l - 1](self._S[l - 1]) *\
                            np.dot(self._weights[l], deltas[l])
        for l in range(self._nLevels):
            dw = np.outer(self._X[l], deltas[l]).squeeze()
            self._weights[l] -= self._eta * dw


    def isDone(self, it, w_old, E_in, maxIters=None, wDiffBound=None,\
                                                     errBound=None):
        '''Check many termination conditions to decide if learning is done.'''
        if maxIters is not None and it >= maxIters:
            return True
        
        wDiff = np.sum([np.linalg.norm(self._weights[l] - w_old[l])\
                        for l in range(self._nLevels)])
        if wDiffBound is not None and wDiff <= wDiffBound:
            return True
        
        if errBound is not None and E_in <= errBound:
            return True
        
        return False

    
    
    def fit(self, X, Y, trackE_in=False, print_stuff=False, **conditions):
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
        w_old = self._weights.copy()
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
            w_old = [layer.copy() for layer in self._weights]

            np.random.shuffle(inds)
            for i in inds:
                x, y = X[i], Y[i]
                self.backPropogate(x, y)
            
            # Update termination relevant variables
            it += 1
            if trackE_in:
                E_ins.append(self.findE_in(X, Y))
                E_in = E_ins[-1]
                if print_stuff:
                    print(f'Epoch {it}: training loss = {E_in}')
            else:
                E_in = None


                
            # Check if to terminate
            if self.isDone(it, w_old, E_in,\
                           maxIters, wDiffBound, errBound):
                return it, np.array(E_ins)
        
    def boundary2D(self):
        if self._nLevels > 1 or self._sizes[0] > 3: 
            raise ValueError('Not a 2D model!')
        return list(self._weights[0][1:]) + [self._weights[0][0]]

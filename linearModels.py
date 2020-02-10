import numpy as np
import matplotlib.pyplot as plt

from MLModels import utils as u



class LinearModel():                                                            
    '''                                                                         
    BaseClass for all LinearModels, uses linear regression as the learning      
    algorithm and mean squared error for the loss function.                     
    '''                                                                         
    def __init__(self, d, weights=None):                                        
        '''                                                                     
        Initialize the weights and number of parameters for a linear model                    
        '''                                                                     
        self._size = d                                                           
        if weights is None:                                                     
            self._weights = np.zeros(d).astype(float)                            
        else:                                                                   
            if len(weights) != d:                                               
                raise ValueError("Initial weight vector of wrong length")       
            self._weights = np.array(weights).astype(float)                                                             


    def signal(self, x):                                                        
        '''                                                                     
        Calculate the signal of a linear model. Vectorized, can calculate       
        a vector of signals for an input matrix.                                
        ''' 
        return np.dot(x, self._weights)                                          
                                                                                
                                                                                
    def __call__(self, x):                                                     
        '''Calculate the hypothesis value for x'''       
        return self.signal(x)                                                 
                                                                                
                                                                                
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
                                                                                
                                                                                
    def fit(self, X, Y, **conditions):                                                      
        '''One step learning using linear regression'''                         
        X_dagger = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)                   
        self._weights = np.dot(X_dagger, Y)
        return 0, np.array([]) # Return nothing to be consistent
        
        
    def boundary2D(self):
        if self._size > 3:
            raise ValueError('Not a 2D model!')
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
        s = self.signal(x)
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
        s = self.signal(x)                                                      
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

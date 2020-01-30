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
        self.size = d                                                           
        if weights is None:                                                     
            self.weights = np.zeros(d).astype(float)                            
        else:                                                                   
            if len(weights) != d:                                               
                raise ValueError("Initial weight vector of wrong length")       
            self.weights = np.array(weights).astype(float)                                                             


    def signal(self, x):                                                        
        '''                                                                     
        Calculate the signal of a linear model. Vectorized, can calculate       
        a vector of signals for an input matrix.                                
        ''' 
        return np.dot(x, self.weights)                                          
                                                                                
                                                                                
    def calculate(self, x):                                                     
        '''Calculate the hypothesis value for x'''       
        return self.signal(x)                                                 
                                                                                
                                                                                
    def err(self, x, y):                                                        
        '''                                                                     
        Find the pointwise loss function, given point and correct output.       
        Vectorized, can calculate a vector of N errors for N inputs and outputs.
        '''
        return (self.calculate(x) - y) ** 2                                     
                                                                                
                                                                                
    def findE_in(self, X, Y):                                                   
        '''                                                                     
        Find the total loss (the in sample error),                              
        given training data with correct outputs                                
        ''' 
        return np.mean(self.err(X, Y))                                          
                                                                                
                                                                                
    def learn(self, X, Y, **conditions):                                                      
        '''One step learning using linear regression'''                         
        X_dagger = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)                   
        self.weights = np.dot(X_dagger, Y)
        return 0, np.array([]) # Return nothing to be consistent
        
        
    def isDone(self, it, w_old, E_in, maxIters=None, wDiffBound=None,\
                                                     errBound=None):
        '''Check many termination conditions to decide if learning is done.'''
        if maxIters is not None and it >= maxIters:
            return True
        
        wDiff = np.linalg.norm(self.weights - w_old)
        if wDiffBound is not None and wDiff <= wDiffBound:
            return True
        
        if errBound is not None and E_in <= errBound:
            return True
        
        return False
        
                                                                  
    def quickPlot(self, X, Y, color='g', label='Hypothesis', axis=None):                            
        '''                                                                     
        For two dimensional input data and corresponding outputs,               
        this will plot the line defined by the weight vector, as well           
        as all the points in the data, to see if the line does separate them    
        as intended                                                             
        '''                                                                     
        if X.shape[1] not in (2, 3):                                            
            raise ValueError('plotting requires 2D input data')                 
                                                                                
        if axis is None:                                                        
            fig, ax = plt.subplots(1, 1)                                        
        else:                                                                   
            ax = axis                                                           
                                                                                
        u.plotLine(*self.weights[1:], self.weights[0], axis=ax,\
                   color=color, label=label)    
        indsPos, indsNeg = (Y == 1), (Y == -1)                                  
        ax.scatter(X[indsPos, -2], X[indsPos, -1], color='b')                   
        ax.scatter(X[indsNeg, -2], X[indsNeg, -1], color='r')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
                                                                                
        if axis is None:                                                        
            return fig, ax



class Perceptron(LinearModel):                                                  
    '''                                                                         
    The Perceptron Model (a binary classifier), uses PLA as the learning        
    algorithm, and binary error as the loss function.                                                                  
    '''
    def calculate(self, x):                                                     
        '''Calculate the hypothesis value for x (sign of the signal)'''         
        s = self.signal(x)
        return np.sign(s)


    def err(self, x, y):                                                        
        '''Binary error'''
        return self.calculate(x) != y 


    def findE_in(self, X, Y):                                                   
        '''                                                                     
        Like the general findE_in for linear models, but keep track of          
        which points in the input are misclassified.                            
        '''                                                                     
        errs = self.err(X, Y)
        self.badInds = np.where(errs)[0]                                        
        return np.mean(errs)                                                    
                                                                                
                                                                                
    def learn(self, X, Y, **conditions):                       
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
        w_old = self.weights.copy()                                         
        E_ins = [self.findE_in(X, Y)]
        
        # Initial check for termination
        if errBound is not None and E_ins[-1] <= errBound:
            return it, np.array(E_ins)
        
        while True:
            w_old = self.weights.copy()
            
            ind = np.random.choice(self.badInds)
            self.weights += Y[ind] * X[ind]
            
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
                                                                                

    def calculate(self, x):                                                     
        '''Calculate the hypothesis value for x (theta of the signal)'''        
        s = self.signal(x)                                                      
        return self.theta(s)
    

    def err(self, x, y):
        '''Cross entropy error'''
        return np.log(1 + np.exp(-y * self.signal(x)))


    def learn(self, X, Y, eta=0.1, useBatch=False, **conditions):       
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
        w_old = self.weights.copy()
        E_ins = [self.findE_in(X, Y)]
        
        inds = np.arange(X.shape[0])
        
        # Initial check for termination
        if errBound is not None and E_ins[-1] <= errBound:
            return it, np.array(E_ins)
                                                           
        while True: 
            w_old = self.weights.copy()
            
            if useBatch:                                                        
                s = -Y * self.signal(X)                                         
                grad = np.dot((-Y * self.theta(s)), X)                          
                self.weights -= eta * grad                                 
            else:                                                               
                np.random.shuffle(inds)                                         
                for i in inds:                                                  
                    x, y = X[i], Y[i]                                           
                    s = - y * self.signal(x)                                    
                    grad = -y * self.theta(s) * x                                    
                    self.weights -= eta * grad
            
            # Update termination relevant variables
            it += 1
            E_ins.append(self.findE_in(X, Y))
            
            # Check if to terminate
            if self.isDone(it, w_old, E_ins[-1],\
                           maxIters, wDiffBound, errBound):
                return it, np.array(E_ins)

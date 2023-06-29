import numpy as np
import matplotlib.pyplot as plt
import os
from   io     import StringIO
import numpy  as np
import math
import sys

import pymc  as pm
import pytensor
import pytensor.tensor as tt

from pdb import set_trace

class MaxEntFire(object):
    """
    Maximum Entropy fire model which takes indepedant variables and coefficants. 
    At the moment, just a linear model fed through a logistic function to convert to 
    burnt area/fire probablity. But we'll adapt that.  
    """ 
    def __init__(self, betas, powers = None, inference = False):
        """
        Sets up the model based on betas and repsonse curve pararameters (response curve 
            not yet implmented
        Arguments:
	    X -- numpy or tensor 2d array of indepenant variables, each columne a different 
                    variable, no. columns (no. variables) is same as length of betas.
	    inference -- boolean.   If True, then used in bayesian inference and uses 
                                        tensor maths. 
			            If False, used in normal mode or prior/posterior sampling 
                                        and uses numpy.
        Returns:
            callable functions. 'fire_model' (below) is the main one to use.
        """
        self.inference = inference
        if self.inference:
            self.numPCK =  __import__('pytensor').tensor
        else:
            self.numPCK =  __import__('numpy')
        
        self.betas = betas
        self.powers = powers

    
    def linear_comination(self, X):
        """calculated predicted burnt area based on indepedant variables. 
            At the moment, just a linear model fed through a logistic function to convert to 
            burnt area/fire probablity. But we'll adapt that.   
        Arguments:
	    X -- numpy or tensor 2d array of indepenant variables, each columne a different 
                    variable, no. columns (no. variables) is same as length of betas.
        Returns:
            numpy or tensor (depdaning on 'inference' option) 1 d array of length equal to 
	    no. rows in X of burnt area/fire probabilities.
        """
        if X is None: sys.exit("Please define X")

        def dot_fun(x, p): return self.numPCK.dot(x, p)
        
        y = dot_fun(X, self.betas)
        
        if self.powers is not None:
            X_powers = self.power_response_curve(X)
            y = y + dot_fun(X_powers, self.powers[0,:])        
        
        return y

    def burnt_area(self, X = None, y = None):
        if y is None: y = self.linear_comination(X)

        BA = 1.0/(1.0 + self.numPCK.exp(-y))
        
        return BA

    def burnt_area_likelihoods(self,
                               burnt_areas_samples = None, #np.linspace(start=0, stop=1, num=101),
                               BA = None, 
                               nesembles = None, *args, **kw):
        if BA is None: BA = self.burnt_area(*args, **kw)
        if burnt_areas_samples is None:
            burnt_areas_samples = np.linspace(start = np.log(1/20000), 
                                              stop = -np.log(1/20000), num = 101)
            burnt_areas_samples = 1.0/(1.0+np.exp(-burnt_areas_samples))

        def prob_y_given_model(y):
            return (BA**y) * ((1-BA)**(1-y))
        
        #BA = 0.33
        probs = np.array([prob_y_given_model(i) for i in burnt_areas_samples])
        
        if nesembles is not None:
            def bootstrap_burntArea(p):
                try:
                    denom =  np.sum(p)
                    if denom == 0: 
                        p[:] = 1.0/len(p)
                        denom = 1.0
                    return np.random.choice(burnt_areas_samples, nesembles, p = p/denom)
                except:
                    yay = p
                    set_trace()
            
            probs = np.apply_along_axis(bootstrap_burntArea, 0, probs)
        
        return probs    


    def hinge_1(x0, y0, a, b):
        """ fits a hinge curve function
        x -- numpy array 
        -- hinge point
        """
    
        if np.all(x1 > x0):
            y = a*x1 + b   
        else:
            y = y0
    
        return y
   

    def hinge_2(self, x, a1, b1, a2, b2):
  
        x0 = (b2-b1)/(a1-a2)
        print("x0 = ", x0)
        y=[]
    
        for xi in x:
            if xi > x0:
                yi = a2*xi + b2   
            else:
                yi = a1*xi + b1
    
            y.append(yi)
        return y

#
    def power_response_curve(self, X):  
        return X**self.powers[1,:]




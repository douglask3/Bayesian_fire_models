import numpy as np
import matplotlib.pyplot as plt
import os
from   io     import StringIO
import numpy  as np
import math

import pymc  as pm
import pytensor
import pytensor.tensor as tt

import sys
sys.path.append('libs/')
from select_key_or_default import *
from pdb import set_trace



class FLAME(object):
    """
    Fogo local Analisado pela Máxima Entropia
    Maximum Entropy fire model which takes independent variables and coefficients. 
    At the moment, just a linear model fed through a logistic function to convert to 
    burnt area/fire probablity. But we'll adapt that.  
    """ 
    def __init__(self, params, inference = False, ConFire = False):
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

        def select_param_or_default(*args, **kw):
            return  select_key_or_default(params, numPCK = self.numPCK, *args, **kw) 
       
        
        self.lin_betas = select_param_or_default('lin_betas', 0.0)
        self.control_betas = select_param_or_default('control_betas', None)
        self.lin_beta_constant = select_param_or_default('lin_beta_constant', 0.0)
        self.pow_betas = select_param_or_default('pow_betas', None)
        self.pow_power = select_param_or_default('pow_power', None)
        self.x2s_betas = select_param_or_default('x2s_betas', None)
        self.x2s_X0    = select_param_or_default('x2s_X0'   , 0.0 )
        self.q = select_param_or_default('q', 0.0)
        self.comb_betas = select_param_or_default('comb_betas', None)   
        self.comb_X0 = select_param_or_default('comb_X0', None) 
        self.comb_p = select_param_or_default('comb_p', None)
        
        #Maria: add your response curve parameter selection thing
        
        if self.control_betas is None:
            self.control_based = False
            self.ConFire = False 
        else:
            self.control_based = True
            self.ConFire = ConFire 
            try:
                self.ncontrols = self.control_betas.shape.eval()[1]
            except:
                self.ncontrols = self.control_betas.shape[1]   

         
    


    def controls(self, Xi):
        #if not self.control_based: return Xi

        def normalize(vector):
            return vector / (self.numPCK.sum(vector**2)**(0.5))
        

        def make_control(X, params):
            params = 2.0 * params - 1.0
            params = normalize(params)
            
            control = self.numPCK.dot(X, params)
            X = X - (self.numPCK.dot(X, params)[:, None] * params)
            
            return control, X

        X = Xi.copy()
        controls = []
        for i in range(self.ncontrols):
            control, X = make_control(X, self.control_betas[:,i])
            controls.append(control)
            
        controls = self.numPCK.transpose(self.numPCK.stack(controls))
        
        return controls

    def burnt_area(self, X, return_controls = False, return_limitations = False):
        """calculated predicted burnt area based on indepedant variables. 
            At the moment, just a linear model fed through a logistic function to convert to 
            burnt area/fire probablity. But we'll adapt that.   
        Arguments:
	    X -- numpy or tensor 2d array of indepenant variables, each columne a different 
                    variable, no. columns (no. variables) is same as length of betas.
        Returns:
            numpy or tensor (depending on 'inference' option) 1 d array of length equal to 
	    no. rows in X of burnt area/fire probabilities.
        """
        self.npoints = X.shape[0]
        if self.control_based:
            self.X_controls = self.controls(X)
        
            if return_controls: return self.X_controls
            def response_curve(i):
                def add_response_curve(Rbetas, FUN, y):
                    if Rbetas is not None:
                        XR = FUN(self.X_controls[:,i], i)
                        y = y + XR * Rbetas[i]
                    return(y)
            
                y = self.X_controls[:,i] * self.lin_betas[i]
                y = add_response_curve(self.pow_betas, self.power_response_curve, y)
                y = add_response_curve(self.x2s_betas, self.X2_response_curve   , y)
                y = add_response_curve(self.comb_betas, self.linear_combined_response_curve , y)
            
                #y = add_response_curve(paramers, function, y)
                # Maria: add yours here 
            
                if self.ConFire:
                    return 1.0/(1.0 + self.numPCK.exp(-y))
                else:
                    return y
            limitations = self.numPCK.stack([response_curve(i) for i in range(self.ncontrols)])
            limitations = self.numPCK.transpose(limitations)
        
            if return_limitations: return limitations
            if self.ConFire:
                BA = self.numPCK.prod(limitations, axis = 1)
            else:
                y = self.numPCK.sum(limitations, axis = 1)
                BA = 1.0/(1.0 + self.numPCK.exp(-y))
            
            return BA
        else:
            y = self.numPCK.dot(X, self.lin_betas)
            
            def add_response_curve(Rbetas, FUN, y):
                if Rbetas is not None:
                    XR = FUN(X)
                    y = y + self.numPCK.dot(XR, Rbetas) 
                return(y)

            y = add_response_curve(self.pow_betas, self.power_response_curve, y)
            y = add_response_curve(self.x2s_betas, self.X2_response_curve   , y)
            y = add_response_curve(self.comb_betas, self.linear_combined_response_curve , y)
        
            # y = add_response_curve(paramers, function, y)
            # Maria: add yours here 

            BA = 1.0/(1.0 + self.numPCK.exp(-y))
            
            return BA


    def burnt_area_no_spread(self, X):
        BA = self.burnt_area(X)
        if self.q == 0.0: return BA
        return BA / (1 + self.q * (1 - BA))

    def burnt_area_spread(self, BA):
        if self.q == 0.0: return BA
        return BA *(1 + self.q) / (BA * self.q + 1)
     
    def selectParams(self, params, i = None):
        if i is None: params = params[i]
        return params

    def power_response_curve(self, X, i = None):
        params = self.selectParams(self.pow_power, i)
        return params**X

    def X2_response_curve(self, X, i):  
        return (X - selectParams(self.x2s_X0, i))**2.0

    def linear_combined_response_curve(self, X, i):
        params = selectParams(self.comb_X0, i)
        return np.log(((np.exp(X - params[i])) ** params[i]) + 1)


import pytensor
import pytensor.tensor as tt
import math  
from pdb import set_trace   
import numpy as np

def ttLogit(x): return tt.log(x/(1-x))

def npLogit(x):
    return np.log(x/(1.0-x))

def npSigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

class zero_inflated_logit(object):

    def obs_given_(Y, fx, sigma, p0, p1):
        '''return tt.sw1itch(
            tt.lt(Y, -150),
            -p0,
            -(1.0 - p0) *(1.0/(sigma * 2.506))*tt.exp(-0.5 * ((Y-fx)/sigma)**2)
        )
        '''
        
        pz = 1.0 - (fx**p1) * (1.0 - p0)
        
        Y = ttLogit(Y)
        fx = ttLogit(fx)

        return tt.switch( tt.lt(Y, -30), 
                          tt.log(pz), 
                          tt.log(1-pz) - tt.log(sigma * tt.sqrt(2*math.pi)) - 
                                ((Y-fx)**2)/(2*sigma**2))
    
    def random_sample_given_central_limit_(mod, sigma, p0, p1): #
        if np.any(mod < 0.0): set_trace()
        
        mod0 = mod.copy()
        
        pz = 1.0 - (mod**p1) * (1.0 - p0)
        
        return mod * (1-pz)

    def random_sample_given_(mod, sigma, p0, p1):
        pz = 1.0 - (mod**p1) * (1.0 - p0)
        mod = npLogit(mod)

        test = (np.random.rand(*pz.shape)) < pz
        
        mod[test] = 0.0
        test = ~test
        
        mod[test] = npSigmoid(np.random.normal(mod[test], sigma)) #(1-pz[test]) * 
        
        return mod
        
    
    def sample_given_(Y, X, sigma, p0, p1):
        
        pz = 1.0 - (X**p1) * (1.0 - p0)
        
        Y[Y < np.exp(-31)] = np.exp(-31)
        Y = npLogit(Y)
        X = npLogit(X)
        test = Y < -30
        
        Y[test]  = pz[test]
        
        test = ~test   
        
        Y[test] = np.exp(-((Y[test]-X[test])**2)/(2*sigma**2))/np.exp(-1.0/(2*sigma**2))#(sigma * np.sqrt(2*math.pi))
        
        return Y

import pytensor
import pytensor.tensor as tt
import math  
from pdb import set_trace   
import numpy as np


class normal_(object):

    def obs_given_(Y, fx, sigma):

        return tt.log(sigma * tt.sqrt(2*math.pi)) - \
                                ((Y-fx)**2)/(2*sigma**2)
    
    def random_sample_given_(mod, sigma):
        
        out =  np.random.normal(mod, sigma)#np.exp(np.log(sigma * np.sqrt(2*math.pi)) - \
                                #((mod)**2)/(2*sigma**2))
        out[out < 0.0] = 0.0
        return out
    
    def sample_given_(Y, X, sigma):
        
        return np.exp(-((Y-X)**2)/(2*sigma**2))/np.exp(-1.0/(2*sigma**2))


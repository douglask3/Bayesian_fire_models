import pytensor
import pytensor.tensor as tt
from pytensor.tensor import TensorVariable
import pymc as pm
from typing import Optional, Tuple
import numpy as np
from pdb import set_trace
import matplotlib.pyplot as plt

def overlap_pred(Y, qSpread):
    if qSpread is  None:
        return Y
    else: return Y *(1 + qSpread) / (Y * qSpread + 1)

def overlap_inverse(Y, qSpread):
    if qSpread is  None:
        return Y
    else: return Y / (1 - Y * qSpread + qSpread)


class MaxEnt(object):
    def __init__(self):
        pass

    #def fire_spread(self,value: TensorVariable, mu: TensorVariable, sigma: TensorVariable,
     #               size: TensorVariable) -> TensorVariable:# value, mu, sigma):
        #set_trace()
        # Conditional logic: if value < fx, return a very low likelihood
    def fire_spread_logp(self, value: TensorVariable, mu: TensorVariable, sigma: TensorVariable, ):
        mu = tt.maximum(mu, 1e-6)  # Avoid division by zero
        sigma = tt.maximum(sigma, 1e-6)  # Ensure sigma isn't too small
        value = tt.clip(value, 1e-6, 1 - 1e-6)
    

        condition = tt.lt(value, mu)
        A = 1.0 / (sigma * mu)

        dist = tt.log((A + 1.0) * tt.pow(1.0 - value, A)/tt.pow(1.0 - value, 1.0 + A))
        #dist = tt.switch(condition, 
        #                 -tt.inf,#0.0, 
        #                 tt.log((A + 1.0) * tt.pow(1.0 - value, A)/tt.pow(1.0 - value, 1.0 + A)))
        return dist

    def fire_spread_random(self, #mu, sigma, size=None):
                           mu: np.ndarray | float,
                           sigma: np.ndarray | float,
                           size : Optional[Tuple[int]]=None,
                          ) -> np.ndarray | float :
        """Generate random samples matching the distribution in logp_fn"""
    
        # Compute A based on sigma and mu
        A = 1.0 / (sigma * mu)
    
        # Sample from a uniform distribution between mu and 1
        u = np.random.uniform(low=0.0, high=(1-mu)**(A+1.0), size=size)
        
        # Apply inverse CDF transformation
        samples = 1 - (u ** (1 / (A + 1)))
        
        # Ensure values are within the correct range
        #samples = np.clip(samples, mu, 1)
        
        return samples
        
    def DensityDistFun(self, Y, fx, CA = None):
        """calculates the log-transformed continuous logit likelihood for Y given fx when Y
            and fx are probabilities between 0-1 with relative areas, CA
            Works with tensor variables.   
        Arguments:
            Y  -- Y  in P(Y|fx). numpy 1-d array
	    fx -- fx in P(Y|fx). tensor 1-d array, length of Y
            qSpread -- parameter that inflates target Y, to account for potential 
                    multi-fire overlap
            CA -- Area for the cover type (cover area). numpy 1-d array, length of Y. 
            Default of None means everything is considered equal area.
        Returns:
            1-d tensor array of liklihoods.
        """    
        fx = tt.switch( tt.lt(fx, 0.00001), 0.00001, fx)
        fx = tt.switch( tt.gt(fx, 0.99999), 0.99999, fx)
        
      
        if CA is not None: 
            prob =  Y*CA*tt.log(fx) + (1.0-Y)*CA*tt.log((1-fx))
        else:
            prob = Y*tt.log(fx) + (1.0-Y)*tt.log((1-fx))
        return prob
    

    def obs_given_(self, fx, Y, CA = None, stochastic = None, qSpread = None):
        if stochastic is not None:
            #set_trace()
            fx = tt.switch( tt.lt(fx, 0.00001), 0.00001, fx)
            fx = tt.switch( tt.gt(fx, 0.99999), 0.99999, fx)
            #fx = pm.Normal("prediction-stochastic", mu=pm.math.logit(fx), 
            #                            sigma = stochastic)

            #fx_stochastic = pm.Deterministic("fx_stochastic", 
            #                                 custom_function(fx, fx, stochastic))
            #fxSt = pm.CustomDist("fxSt", 
            #         fx, 
            #         stochastic, dist=self.fire_spread)
            #fxST = pm.Deterministic("fxST", self.fire_spread(fx, stochastic))
            #yay1 = self.fire_spread_random(0.5, 1, 10000)
            ##yay2 = self.fire_spread_random(0.8, 1, 10000)
            #yay3= self.fire_spread_random(0.8, 2, 10000)
            #yay4 = self.fire_spread_random(0.8, 0.5, 10000)
            fx = pm.CustomDist("fxST",
                               fx, stochastic,
                               logp = self.fire_spread_logp, 
                               random=self.fire_spread_random,
                               size=Y.shape[0])
            set_trace()
            #fx = pm.Deterministic("fxST", self.fire_spread(fx, n_samples=20))                           
            #fx = pm.CustomDist( "fxST", fx, stochastic, 
            #                     logp=self.fire_spread)
            #fxST = pm.CustomDist("fxST", fx, stochastic, self.fire_spread, dist = fire_spread)
            #set_trace()
            #fx = pm.math.sigmoid(fx)
        
        Y = overlap_pred(Y, qSpread)
        if CA is None:
            error = pm.DensityDist("error", fx,
                                   logp = self.DensityDistFun, 
                                   observed = Y)
        else:  
            error = pm.DensityDist("error", fx, CA,
                                   logp = self.DensityDistFun, 
                                   observed = Y)
        
        return error
            
    def random_sample_given_central_limit_(self, mod, qSpread = None, CA = None): #
        #return mod
        return overlap_inverse(mod, qSpread)

    def random_sample_given_(self, mod, qSpread = None, CA = None):
        #return mod
        return overlap_inverse(mod, qSpread)
    
    def sample_given_(self, Y, X, *args, **kw):
        X1 = 1 - X
        def prob_fun(y):
            return (y**X) * ((1-y)**X1)
        prob = prob_fun(Y)/prob_fun(X)
        return prob
    


import pytensor
import pytensor.tensor as tt
import pymc as pm
from pdb import set_trace

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
        fx = tt.switch( tt.lt(fx, 0.0000000001), 0.0000000001, fx)
        fx = tt.switch( tt.gt(fx, 0.9999999999), 0.9999999999, fx)
        
      
        if CA is not None: 
            prob =  Y*CA*tt.log(fx) + (1.0-Y)*CA*tt.log((1-fx))
        else:
            prob = Y*tt.log(fx) + (1.0-Y)*tt.log((1-fx))
        return prob
    

    def obs_given_(self, fx, Y, CA = None, stochastic = None, qSpread = None):
        if stochastic is not None:
            fx = pm.Normal("prediction-stochastic", mu=pm.math.logit(fx), 
                                        sigma = stochastic) 
            fx = pm.math.sigmoid(fx)
        #set_trace()
        
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
    


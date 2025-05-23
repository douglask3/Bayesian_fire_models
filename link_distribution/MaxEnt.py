import pytensor
import pytensor.tensor as tt
from pytensor.tensor import gammaln
from pytensor.tensor import TensorVariable
import pymc as pm
from typing import Optional, Tuple
import numpy as np
from pdb import set_trace
import matplotlib.pyplot as plt

def any_in(list_str, string):
    return any(np.array([string in item for item in list_str]))

def element_ref(list_v, list_names, string):
    return [item for name, item in zip(list_names, list_v) if string in name]

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
        
    def DensityDistFun(self, Y, fx, Ncells, qSpread = 1.0, detection_epslion = 0.0, CA = None):
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
        fx = tt.switch( tt.lt(fx, 0.00000000000001), 0.00000000000001, fx)
        fx = tt.switch( tt.gt(fx, 0.99999999999999), 0.99999999999999, fx)
        
        Y = (Y + detection_epslion)/(1.0 + detection_epslion)
        Y = overlap_pred(Y, qSpread)
        #set_trace()
        if CA is not None: 
            prob =  Y*CA*tt.log(fx) + (1.0-Y)*CA*tt.log((1-fx))
        else:
            prob = Y*tt.log(fx) + (1.0-Y)*tt.log((1-fx))

        mean_fx = tt.mean(fx)
        mean_y = tt.mean(Y)
        k = Ncells/2.0 # 1.0/tt.var(Y)#
        epsilon = 0.000000000001
        alpha = mean_y * k + epsilon
        beta = (1 - mean_y) * k + epsilon
        logp_global = ((alpha - 1) * tt.log(mean_fx)+ (beta - 1) * tt.log(1 - mean_fx) \
                       - (gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)))
        #logp_global = mean_y * tt.log(mean_fx) + (1 - mean_y) * tt.log(1 - mean_fx)
        return prob + logp_global/Ncells #tt.sum(prob)
    
    def define_qSpread_param(self, params, param_names, inference = True, sigma = None):
        #set_trace()
        if any_in(param_names, 'qSpread_mu'):
            mu = element_ref(params, param_names, 'qSpread_mu')[0]
            if sigma is None:
                sigma = element_ref(params, param_names, 'qSpread_sigma')[0]
            if sigma == 0.0:
                qSpread = mu
            else:
                if inference:
                    qSpread = pm.LogNormal("qSpread", mu = mu, sigma = sigma)
                else:
                    qSpread = np.random.lognormal(mu, sigma, 1)
        elif any_in(param_names, 'qSpread'):
            qSpread =  element_ref(params, param_names, 'qSpread')[0]
        else:
            qSpread = 1.0
        return qSpread
    
    def define_detection_efficency_param(self, params, param_names, 
                                         inference = True, determined = False):
        #set_trace()

        if any_in(param_names, 'detection_alpha'):
            alpha = element_ref(params, param_names, 'detection_alpha')[0]
            beta  = element_ref(params, param_names, 'detection_beta' )[0]
            if determined:
                epsilon = alpha/(alpha + beta)
            else:
                if inference:
                    epsilon = pm.Beta("epsilon", alpha = alpha, beta = beta)
                else:
                    epsilon = np.random.beta(alpha, beta, 1)
        elif any_in(param_names, 'detection'):
            epsilon =  element_ref(params, param_names, 'detection')[0]
        else:
            epsilon = 0.0
        return epsilon
    
    def obs_given_(self, fx, Y, CA = None, params = None):
        if params is not None:
            param_names = [param.name[5:] for param in params]
            qSpread = self.define_qSpread_param(params, param_names)
            detection_epslion = self.define_detection_efficency_param(params, param_names)
            
            #mean_pred = tt.mean(fx)
            if any_in(param_names, 'stochastic'):
                #set_trace()
                fx = tt.switch( tt.lt(fx, 0.00000000000001), 0.00000000000001, fx)
                fx = tt.switch( tt.gt(fx, 0.99999999999999), 0.99999999999999, fx)
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
                #fx = pm.CustomDist("fxST",
                #                   fx, stochastic,
                #                   logp = self.fire_spread_logp, 
                #                   random=self.fire_spread_random,
                #                   size=Y.shape[0])
                #set_trace()
                #fx = pm.Deterministic("fxST", self.fire_spread(fx, n_samples=20))                           
                #fx = pm.CustomDist( "fxST", fx, stochastic, 
                #                     logp=self.fire_spread)
                #fxST = pm.CustomDist("fxST", fx, stochastic, self.fire_spread, dist = fire_spread)
                #set_trace()
                #fx = pm.math.sigmoid(fx)
            
           # Y = overlap_pred(Y, qSpread)
        
        if CA is None:
            error = pm.DensityDist("error", fx, len(Y), qSpread, detection_epslion,
                                   logp = self.DensityDistFun, 
                                   observed = Y)
        else:  
            error = pm.DensityDist("error", fx, len(Y), qSpread, detection_epslion, CA,
                                   logp = self.DensityDistFun, 
                                   observed = Y)
        
        #penalty = tt.switch(mean_pred < 0.0000001, -1e8, 0.0)  # discourage implausibly low burn

        # Apply the penalty
        #pm.Potential("burning_not_too_low", penalty)
        return error
            
    def random_sample_given_central_limit_(self, mod, params = None, CA = None): #
        return mod
        param_names = params.keys()
        params = params.values()
        qSpread = self.define_qSpread_param(params, param_names, False, 0.0)        
        return overlap_inverse(mod, qSpread)

    def random_sample_given_(self, mod, params = None, CA = None):
        #return mod
        param_names = params.keys()
        params = params.values()
        qSpread = self.define_qSpread_param(params, param_names, False)
        detection_epslion = self.define_detection_efficency_param(params, param_names, False)
        #set_trace()
        return overlap_inverse(mod, qSpread)
    
    def sample_given_(self, Y, X, *args, **kw):
        X1 = 1 - X
        def prob_fun(y):
            return (y**X) * ((1-y)**X1)
        prob = prob_fun(Y)/prob_fun(X)
        return prob
    


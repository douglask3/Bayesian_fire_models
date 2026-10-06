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

def any_in(list_str, string):
    return any(np.array([string in item for item in list_str]))

def element_ref(list_v, list_names, string):
    out =  [item for name, item in zip(list_names, list_v) if string in name]
    if len(out) == 1:
        out = out[0]
    return out
    
def select_param(params, pname):
    
    try:
        param_names = [param.name[5:] for param in params]
    except:
        param_names = [i for i in params.keys()]
        params = [params[name] for name in param_names]
    return element_ref(params, param_names, pname) 
    

class zero_inflated_logit(object):
    def __init__(self, data_store = None, ensemble_member = None, common_noise = False, eg_cube = None, lmask = None):
        self.data_store = data_store
        self.ensemble_member = ensemble_member
        self.common_noise = common_noise

        if eg_cube is not None and lmask is not None:
            flat_idx = np.where(lmask)[0]
            ntime, nlat, nlon = eg_cube.shape
            t_idx, lat_idx, lon_idx = np.unravel_index(flat_idx, (ntime, nlat, nlon))
            
            time_coord = eg_cube.coord('time').points
            lat_coord = eg_cube.coord('latitude').points
            lon_coord = eg_cube.coord('longitude').points
            
            self.times = time_coord[t_idx]
            lats = lat_coord[lat_idx]
            lons = lon_coord[lon_idx]
            self.lats = np.round(lats, 5)
            self.lons = np.round(lons, 5)
        pass
    
    def get_stocastic_params(self, params):
        sigma = select_param(params, 'sigma')
        p0 = select_param(params, 'p0')
        p1 = select_param(params, 'p1')
        return sigma, p0, p1
    

    def obs_given_(self, fx, Y, CA = None, params = None):#, sigma, p0, p1):
        '''return tt.sw1itch(
            tt.lt(Y, -150),
            -p0,
            -(1.0 - p0) *(1.0/(sigma * 2.506))*tt.exp(-0.5 * ((Y-fx)/sigma)**2)
        )
        '''
         
        sigma, p0, p1 = self.get_stocastic_params(params)
        pz = 1.0 - (fx**p1) * (1.0 - p0)
        
        Y = ttLogit(Y)
        fx = ttLogit(fx)

        return tt.switch( tt.lt(Y, -30), 
                          tt.log(pz), 
                          tt.log(1-pz) - tt.log(sigma * tt.sqrt(2*math.pi)) - 
                                ((Y-fx)**2)/(2*sigma**2))
    
    def random_sample_given_central_limit_(self, mod, sigma, params = None, CA = None): #
        if np.any(mod < 0.0): set_trace()
        sigma, p0, p1 = self.get_stocastic_params(params)
        mod0 = mod.copy()
        
        pz = 1.0 - (mod**p1) * (1.0 - p0)
        
        return mod * (1-pz)

    def random_sample_given_(self, mod, params = None, CA = None):
        sigma, p0, p1 = self.get_stocastic_params(params)
        pz = 1.0 - (mod**p1) * (1.0 - p0)
        mod = npLogit(mod)

        test = (np.random.rand(*pz.shape)) < pz
        
        mod[test] = 0.0
        test = ~test
        
        mod[test] = npSigmoid(np.random.normal(mod[test], sigma)) #(1-pz[test]) * 
        
        return mod
        

    def sample_given_(self, Y, X, params = None, CA = None):
        
        sigma, p0, p1 = self.get_stocastic_params(params)
        
        pz = 1.0 - (X**p1) * (1.0 - p0)
        
        Y[Y < np.exp(-31)] = np.exp(-31)
        Y = npLogit(Y)
        X = npLogit(X)
        test = Y < -30
        
        Y[test]  = pz[test]
        
        test = ~test   
        
        Y[test] = np.exp(-((Y[test]-X[test])**2)/(2*sigma**2))/np.exp(-1.0/(2*sigma**2))#(sigma * np.sqrt(2*math.pi))
        
        return Y

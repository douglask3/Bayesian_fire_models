import arviz as az
import numpy as np
import iris

import os
import sys
sys.path.append('fire_model/')
sys.path.append('libs/')

from combine_path_and_make_dir import * 


from FLAME import FLAME

from iris_plus import insert_data_into_cube

import pytensor
import pytensor.tensor as tt

from pdb import set_trace

def select_post_param(trace):
    def select_post_param_name(name): 
        out = trace.posterior[name].values
        A = out.shape[0]
        B = out.shape[1]
        new_shape = ((A * B), *out.shape[2:])
        return np.reshape(out, new_shape)

    params = trace.to_dict()['posterior']
    params_names = params.keys()
    params = [select_post_param_name(var) for var in params_names]
    return params, [var for var in params_names]

def logistic_probability_tt(Y, fx, qSpread = None, penalty = 1.0, CA = None):
    """calculates the log-transformed continuous logit likelihood for Y given fx when Y
       and fx are probabilities between 0-1 with relative areas, CA
       Works with tensor variables.   
    Arguments:
        Y  -- Y  in P(Y|fx). numpy 1-d array
	fx -- fx in P(Y|fx). tensor 1-d array, length of Y
        CA -- Area for the cover type (cover area). numpy 1-d array, length of Y. Default of None means everything is considered equal area.
    Returns:
        1-d tensor array of liklihoods.
        
    """
    
    fx = tt.switch( tt.lt(fx, 0.0000000001), 0.0000000001, fx)
    fx = tt.switch( tt.gt(fx, 0.9999999999), 0.9999999999, fx)
    
    if qSpread is not None:
        Y = Y *(1 + qSpread) / (Y * qSpread + 1)
      
    if CA is not None: 
        prob =  Y*CA*tt.log(fx) + (1.0-Y)*CA*tt.log((1-fx))
    else:
        prob = Y*tt.log(fx) + (1.0-Y)*tt.log((1-fx))
    
    prob = prob - 0.01* tt.log(penalty)
    
    return prob

def logistic_how_likely(Y, X):
    
    X1 = 1 - X
    def prob_fun(y):
        return (y**X) * ((1-y)**X1)
    prob = prob_fun(Y)/prob_fun(X)
    return prob
    

def runSim_MaxEntFire(trace, sample_for_plot, X, eg_cube, lmask, run_name, 
                      dir_samples, grab_old_trace, 
                      class_object = FLAME, method = 'burnt_area',
                      test_eg_cube = False, out_index = None, *args, **kw):  
    
    def sample_model(i, run_name = 'control'):   
        dir_sample =  combine_path_and_make_dir(dir_samples, run_name)
        file_sample = dir_sample + '/sample-pred' + str(i) + '.nc'

        dont_do_prob = True
        if test_eg_cube:
            file_prob = dir_sample + '/sample-prob' + str(i) + '.nc'
            if grab_old_trace  and os.path.isfile(file_prob):
                prob = iris.load_cube(file_prob)
            else:
                dont_do_prob = False
            
        if grab_old_trace and os.path.isfile(file_sample) and dont_do_prob:
            out = iris.load_cube(file_sample)
            if test_eg_cube:           
                return out, prob
            else:
                return out
        
        coord = iris.coords.DimCoord(i, "realization")
        def make_into_cube(dat, filename):            
            dat = insert_data_into_cube(dat, eg_cube, lmask)
            dat.add_aux_coord(coord)
            iris.save(dat, filename)
            return dat

        print("Generating Sample:" + file_sample)
        param_in = [param[i] if param.ndim == 1 else param[i,:] for param in params]
        param_in = dict(zip(params_names, param_in))
        obj = class_object(param_in)
        out = getattr(obj, method)(X, *args, **kw)
        
        if out_index is not None: out = out[:, out_index]
        if test_eg_cube: 
            prob = logistic_how_likely(eg_cube.data.flatten()[lmask], out)
            prob = make_into_cube(prob, file_prob)       
        out = make_into_cube(out, file_sample)

        if test_eg_cube: 
            return out, prob
        else:
            return out
        

    params, params_names = select_post_param(trace)  
    nits = len(trace.posterior.chain)*len(trace.posterior.draw)
    idx = range(0, nits, int(np.floor(nits/sample_for_plot)))
    out = np.array(list(map(lambda id: sample_model(id, run_name), idx)))
    
    if test_eg_cube: 
        prob = iris.cube.CubeList(out[:,1]).merge_cube()
        prob = prob.collapsed('realization', iris.analysis.MEAN)
        return iris.cube.CubeList(out[:,0]).merge_cube(), prob
    else:
        try:
            return iris.cube.CubeList(out).merge_cube()
        except:
            out = np.array(list(map(lambda id: sample_model(id, run_name), idx)))
            return iris.cube.CubeList(out).merge_cube()
        


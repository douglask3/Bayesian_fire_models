import arviz as az
import numpy as np
import pymc as pm
import iris

import os
import sys
sys.path.append('fire_model/')
sys.path.append('libs/')
sys.path.append('link_distribution/')

from combine_path_and_make_dir import * 

from FLAME import FLAME
from ConFire import ConFire
from MaxEnt import MaxEnt
from zero_inflated_logit import zero_inflated_logit
from normal_ import  normal_

from iris_plus import insert_data_into_cube

import pytensor
import pytensor.tensor as tt

from pdb import set_trace

import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.pyplot as plt

def select_post_param(trace):
    """Selects paramaeters from a pymc nc trace file.   
    Arguments:
        trace -- pymc netcdf trace file
    Returns:
        dict of paramater values with each item names after the parameter        
    """

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

def contruct_param_comb(i, params, params_names, extra_params):
    param_in = [param[i] if param.ndim == 1 else param[i,:] for param in params]
    param_in = dict(zip(params_names, param_in))
    param_in.update(extra_params)
    return param_in

def runSim_MaxEntFire(trace, sample_for_plot, X, eg_cube, lmask, run_name, 
                      dir_samples, grab_old_trace, extra_params = None,
                      class_object = FLAME, method = 'burnt_area',
                      link_func_class = MaxEnt, hyper = True, sample_error = True,
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

        param_in = contruct_param_comb(i, params, params_names, extra_params)
        link_param_in = {key: value for key, value in param_in.items() \
                       if key.startswith('link-')}

        obj = class_object(param_in)
        out = getattr(obj, method)(X, *args, **kw)
        
    
        if out_index is not None: out = out[:, out_index]
        if test_eg_cube:
            prob = link_func_class.sample_given_(eg_cube.data.flatten()[lmask], out, 
                                                   *link_param_in.values())
            
            prob = make_into_cube(prob, file_prob) 
        
        
        if hyper:
            if sample_error:
                out = link_func_class.random_sample_given_(out, *link_param_in.values()) 
            else:
                out = link_func_class.random_sample_given_central_limit_(out, 
                                                                    *link_param_in.values()) 
                     
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
        mout = out[:, 1]
        for cube in mout:
            if cube.coords('month'):
                cube.coord('month').mask = False
            if cube.coords('month_number'):
                cube.coord('month_number').mask = False
            #cube.coord('month').points = mout[0].coord('month').points
            #cube.coord('month_number').points = mout[0].coord('month').points
        try:
            prob = iris.cube.CubeList(mout).merge_cube()
        except:
            set_trace()
        
        prob = prob.collapsed('realization', iris.analysis.MEAN)
        return iris.cube.CubeList(out[:,0]).merge_cube(), prob
    else:
        try:
            return iris.cube.CubeList(out).merge_cube()
        except:
            out = np.array(list(map(lambda id: sample_model(id, run_name), idx)))
            return iris.cube.CubeList(out).merge_cube()
        
def get_step_method(step_type):
    # Dictionary mapping strings to step methods
    step_methods = {
        'nuts': pm.NUTS,
        'hmc': pm.HamiltonianMC,
        'metropolis': pm.Metropolis,
        'binary_metropolis': pm.BinaryMetropolis,
        'binary_gibbs_metropolis': pm.BinaryGibbsMetropolis,
        'categorical_gibbs_metropolis': pm.CategoricalGibbsMetropolis,
        'demetropolis': pm.DEMetropolis,
        'demetropolis_z': pm.DEMetropolisZ,
        'slice': pm.Slice,
        'compound_step': pm.CompoundStep,  # Requires list of methods
        # Proposal distributions (used in conjunction with Metropolis-like samplers)
        'cauchy_proposal': pm.CauchyProposal,
        'laplace_proposal': pm.LaplaceProposal,
        'multivariate_normal_proposal': pm.MultivariateNormalProposal,
        'normal_proposal': pm.NormalProposal,
        'poisson_proposal': pm.PoissonProposal,
        'uniform_proposal': pm.UniformProposal,
    }

    # Return the appropriate step method function, or raise an error if invalid
    if step_type.lower() in step_methods:
        return step_methods[step_type.lower()]
    else:
        raise ValueError(f"Unknown step method: {step_type}")

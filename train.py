import sys
sys.path.append('fire_model/')
sys.path.append('libs/')

from FLAME import FLAME

from read_variable_from_netcdf import *
from combine_path_and_make_dir import * 
from namelist_functions import *
from pymc_extras import *

import os
from   io     import StringIO
import numpy  as np
import pandas as pd
import math

import pymc  as pm
import arviz as az


def fit_MaxEnt_probs_to_data(Y, X, CA = None, niterations = 100, priors = None, *arg, **kw):
    """ Bayesian inerence routine that fits independant variables, X, to dependant, Y.
        Based on the MaxEnt solution of probabilities. 
    Arguments:
        Y-- dependant variable as numpy 1d array
	X -- numpy 2d array of indepenant variables, each columne a different variable
        CA -- Area for the cover type (cover area). None means doesnt run otherwise, numpy 1-d array, length of Y. Defalt to None.
	niterations -- number of iterations per chain when sampling the postior during 
                NUTS inference. Default of 100.
	*args, **kw -- arguemts passed to 'pymc.sample'

    Returns:
        pymc traces, returned and saved to [out_dir]/[filneame]-[metadata].nc
    """

    trace_callback = None
    try:
        if "SLURM_JOB_ID" in os.environ:
            def trace_callback(trace, draw):        
                if len(trace) % 10 == 0:
                    print('chain' + str(draw[0]))
                    print('trace' + str(len(trace)))
    except:
        pass        

    try:
        Y = Y.data
    except:
        pass
    with pm.Model() as max_ent_model:
        ## set priors
        nvars = X.shape[1]
        if priors is None:
            priors = {"q":     pm.LogNormal('q', mu = 0.0, sigma = 1.0),
                     "lin_beta_constant": pm.Normal('lin_beta_constant', mu = 0, sigma = 100),
                     "lin_betas": pm.Normal('lin_betas', mu = 0, sigma = 100, shape = nvars),
                     "pow_betas": pm.Normal('pow_betas', mu = 0, sigma = 100, shape = nvars),
                     "pow_power": pm.LogNormal('pow_power', mu = 0, sigma = 1, shape = nvars),
                     "x2s_betas": pm.Normal('x2s_betas', mu = 0, sigma = 100, shape = nvars),
                     "x2s_X0"   : pm.Normal('x2s_X0'   , mu = 0, sigma = 1, shape = nvars),
                     "comb_betas": pm.Normal('comb_betas', mu = 0, sigma = 100, shape = nvars),
                     "comb_X0": pm.Normal('comb_X0', mu = 0.5, sigma = 1, shape = nvars),
                     "comb_p": pm.Normal('comb_p', mu = 0, sigma = 1 , shape = nvars)
                     }
        else:
            def define_prior(prior, pn):
                kws = prior.copy()
                kws.pop('pname')
                kws.pop('np')
                kws.pop('dist')
                shape = prior['np']
                if shape == 'nvars': 
                    shape = nvars 
                elif shape == 'nvars*2':
                    shape = (nvars, nvars)
                
                return getattr(pm, prior['dist'])(prior['pname'] + str(pn), 
                                      shape = shape, **kws)

            priors_names =[prior['pname'] for prior in priors]

            count_priors = [priors_names[:i].count(string) \
                            for i, string in enumerate(priors_names)]
            
            priors = [define_prior(prior, pn) for prior, pn in zip(priors, count_priors)]
            
            # Create a dictionary to store items based on the first list
            grouped_items = {}
            for idx, string in enumerate(priors_names):
                if string not in grouped_items:
                    grouped_items[string] = []
                grouped_items[string].append(priors[idx])

            # Convert the dictionary values to a list
            result_list = list(grouped_items.values())
            
            priors_names = list(dict.fromkeys(priors_names))
            priors = {priors_names[idx]: item[0] if len(item) == 1 else item \
                      for idx, item in enumerate(result_list)}
            
        
        ## run model
        model = FLAME(priors, inference = True)
        prediction = model.burnt_area(X)  
        
        ## define error measurement
        if CA is None:
            error = pm.DensityDist("error", prediction, priors['q'], 
                                   logp = logistic_probability_tt, 
                                   observed = Y)
        else:
            CA = CA.data
            error = pm.DensityDist("error", prediction, priors['q'], CA, 
                                   logp = logistic_probability_tt, 
                                   observed = Y)
                
        ## sample model
        trace = pm.sample(niterations, return_inferencedata=True, 
                          callback = trace_callback, *arg, **kw)
        attempts = 1
        while attempts <= 10:
            try:
                trace = pm.sample(niterations, return_inferencedata=True, 
                                  callback = trace_callback, *arg, **kw)
                attempts = 100
            except:
                print("sampling attempt " + str(attempts) + " failed. Trying a max of 10 times")
                attempts += 1
    return trace


def train_MaxEnt_model_from_namelist(namelist = None, **kwargs):

    variables = read_variable_from_namelist_with_overwite(namelist, **kwargs)
    
    variables['filename_out'] = \
              '_'.join([file[:-3] for file in variables['x_filen_list']]) + \
              '-frac_points_' + str(variables['fraction_data_for_sample'])
    
    if 'dir' not in variables and 'dir_training' in variables:
        variables['dir'] = variables['dir_training']

    
    return train_MaxEnt_model(**variables)



def train_MaxEnt_model(y_filen, x_filen_list, CA_filen = None, priors = None,
                       dir = '', filename_out = '',
                       dir_outputs = '',
                       fraction_data_for_sample = 1.0,
                       subset_function = None, subset_function_args = None,
                       niterations = 100, cores = 4, model_title = 'no_name',
                       subfolder = '', 
                       grab_old_trace = False, **kws):
                       
    ''' Opens up training data and trains and saves Bayesian Inference optimization of model. 
        see 'fit_MaxEnt_probs_to_data' for details how.
    Arguments:
	y_filen -- filename of dependant variable (i.e burnt area)
        x_filen_list -- filanames of independant variables
            (ie bioclimate, landscape metrics etc)
        filename_out -- string of the start of the traces output name. Detault is blank. 
		Some metadata will be saved in the filename, so even blank will 
                save a file.
        dir_outputs --string of path to output location. This is where the traces netcdf file 
                will be saved.
        fraction_data_for_sample -- fraction of gridcells used for optimization
	subset_function -- a list of constrain function useful for constraining and resticting 
                data to spatial locations and time periods/months. Default is not to 
                constrain (i.e "None" for no functions")
        subset_function_args -- list of arguements that feed into subset_function
        niterations -- number of iterations or samples )after warm-up) in optimixation for each
                chain. Equilivent to number of ensemble members.
        cores - how many chains to start (confusiong name, I know).
                When running on slurm, also number of cores
        model_title - title of model run. A str default to 'no_name'. Used to initially to name 
                the dir everythings stored in
	grab_old_trace -- Boolean. If True, and a filename starting with 'filename' and 
                containing some of the same setting (saved in filename) exists,  it will open 
                and return this rather than run a new one. Not all settings are saved for 
                identifiation, so if in doubt, set to 'False'.
    Returns:
        pymc traces, returned and saved to [out_dir]/[filneame]-[metadata].nc and the scalers
        used on independant data to normalise it, useful for predicting model
    '''
    
    print("====================")
    print("Optimization started")
    print("====================")
    dir_outputs = combine_path_and_make_dir(dir_outputs, model_title)
    dir_outputs = combine_path_and_make_dir(dir_outputs, subfolder)
    out_file =   filename_out + '-nvariables_' + \
                 '-frac_random_sample' + str(fraction_data_for_sample) + \
                 '-nvars_' +  str(len(x_filen_list)) + \
                 '-niterations_' + str(niterations * cores)
    
    data_file = dir_outputs + '/data-'   + out_file + '.nc'
    trace_file = dir_outputs + '/trace-'   + out_file + '.nc'
    scale_file = dir_outputs + '/scalers-' + out_file + '.csv'
    
    
    ## check if trace file exsts and return if wanted
    if os.path.isfile(trace_file) and os.path.isfile(scale_file) and grab_old_trace:
        print("======================")
        print("Old optimization found")
        print("======================")
        trace = az.from_netcdf(trace_file)
        scalers = pd.read_csv(scale_file).values   
    else:
        print("opening data for inference")
        
        common_args = {'y_filename': y_filen,
            'x_filename_list': x_filen_list,
            'add_1s_columne': False,
            'dir': dir,
            'x_normalise01': True,
            'frac_random_sample': fraction_data_for_sample,
            'subset_function': subset_function,
            'subset_function_args': subset_function_args,
            **kws
        }
        
        if CA_filen is not None:
            # Process CA_filen when it is provided
            Y, X, CA, lmask, scalers = read_all_data_from_netcdf(CA_filename = CA_filen, 
                                                                 **common_args)
            CA = CA/np.max(CA)
        else:
            Y, X, lmask, scalers = read_all_data_from_netcdf(**common_args)
            CA = None
        
        if np.min(Y) < 0.0 or np.max(Y) > 100:
            print("target variable does not meet expected unit range " + \
                  "(i.e, data poimts should be fractions, but values found less than " + \
                  "0 or greater than 1)")
            if np.mean(Y>1.0) > 0.01:
                sys.exit() 
            else:
                Y[Y>1.0] = 0.9999999999999
        if np.min(Y) > 1.0:
            if np.max(Y) < 50:
                print("WARNING: target variable has values greater than 1 all less than 50." + \
                      " Interpreting at a percentage, but you should check")
            Y = Y / 100.0
    
        print("======================")
        print("Running trace")
        print("======================")
        trace = fit_MaxEnt_probs_to_data(Y, X, CA = CA,  niterations = niterations, 
                                         cores = cores, priors = priors)
    
        ## save trace file
        trace.to_netcdf(trace_file)
        pd.DataFrame(scalers).to_csv(scale_file, index = False)

        print("=====================")
        print("Optimization complete")
        print("=====================")


    # Save info to namelist.
    variable_info_file = dir_outputs + 'variables_info-' + out_file + '.txt'
    desired_variable_names = ["dir_outputs", "filename_out",
                              "out_file", "data_file", 
                              "trace_file", "scale_file", 
                              "dir", "y_filen", "x_filen_list", "CA_filen",
                              "subset_function", "subset_function_args", 
                              "trace_file", "scale_file"]

    # Create a dictionary of desired variables and their values
    variables_to_save = {name: value for name, value in locals().items() \
                            if name in desired_variable_names}
    
    write_variables_to_namelist(variables_to_save, variable_info_file)

    print("\ntrace filename:\n\t" + trace_file)
    print("\nscalers filename:\n\t" + scale_file)
    print("\nall information writen to namelist:\n\t" + variable_info_file)
 
    return trace, scalers, variable_info_file


if __name__=="__main__":
    """ Running optimization. 
    Variables that need setting:
        model_title -- name of model run. Used as directory and filename.
        dir_training -- The directory of the training data inclduing 
            dependant and independant variables
        y_filen -- filename of dependant variable (i.e burnt area)
        x_filen_list -- filanames of independant variables
            (ie bioclimate, landscape metrics etc)
        Priors -- A list of priors. Each prior goes:
             {pname: 'parameter name', np: no. parameters, \
              dist: 'pymc distribution name', **distribution setting}
            Where:
                - parameter name is a string that needs to match the parameters in the 
                  fire model (i.e, in FLAME)
                - no. of paramaters is either numierci, and is the amount of times you use 
                  that parameter, or 'nvars', which means use as many parameters as variables.
                - 'pymc distribution name' - pymc distribution or a string that matches one 
                  of the distributions in pymc. See list: 
                        https://www.pymc.io/projects/docs/en/stable/api/distributions.html
                - **distribution setting are additional inputs it the distribution   
        cores - how many chains to start (confusiong name, I know).
            When running on slurm, also number of cores
        fraction_data_for_sample -- fraction of gridcells used for optimization
        niterations -- number of iterations or samples )after warm-up) in optimixation for each
            chain. Equilivent to number of ensemble members.
        months_of_year --- which months to extact on training and projecting
        grab_old_trace -- Boolean. If True and there's an appripritate looking old trace file, 
            then  optimisation is skipped that file is loaded instead. 
            This isn't totally infalable, so if doing a final run and in doubt, set to False
    Returns:
        outputs trace file and info (variable scalers) needed for evaluation and projection.
    """ 
    """ 
        EXAMPLE 1 - namelist
    """

    namelist = "namelists//simple_example_namelist.txt"

    train_MaxEnt_model_from_namelist('namelists/simple_example.txt')
    set_trace()
    """ 
        EXAMPLE 2 - python code 
    """
    ### input data paths and filenames
    model_title = 'train_from_bottom-biome-all-lin_pow-PropSpread2'
    dir_training = "../ConFIRE_attribute/isimip3a/driving_data/GSWP3-W5E5-20yrs/Brazil/AllConFire_2000_2009/"
    y_filen = "GFED4.1s_Burned_Fraction.nc"
    CA_filen = None
    
    x_filen_list=["road_density.nc","trees.nc","consec_dry_mean.nc",
                  "crop.nc", "pas.nc",  "savanna.nc", "grassland.nc"] 

    x_filen_list= ["ed.nc", "consec_dry_mean.nc", "savanna.nc", "cveg.nc", "rhumid.nc",
                   "lightn.nc", "popDens.nc", "forest.nc", "precip.nc",
                   "pasture.nc", "cropland.nc", "grassland.nc", #"np.nc",
                   "tas_max.nc", "mpa.nc", # "tca.nc",, "te.nc", "tas_mean.nc"
                   "vpd.nc", "soilM.nc"]

    priors =  [{'pname': "q",'np': 1, 'dist': 'LogNormal', 'mu': 0.0, 'sigma': 1.0},
               {'pname': "lin_beta_constant",'np': 1, 'dist': 'Normal', 'mu': 0.0, 'sigma': 100},
               {'pname': "lin_betas",'np': 'nvars', 'dist': 'Normal', 'mu': 0.0, 'sigma': 100},
               {'pname': "pow_betas",'np': 'nvars', 'dist': 'Normal', 'mu': 0.0, 'sigma': 100},
               {'pname': "pow_power",'np': 'nvars', 'dist': 'LogNormal', 'mu': 0.0, 'sigma': 1}]

    ### optimization info
    niterations = 400
    cores = 1
    fraction_data_for_sample = 0.005
    min_data_points_for_sample = 500
    months_of_year = [7]
    year_range = [2002, 2009]
    biome_ID = 0

    subset_function = [sub_year_range, 
                       sub_year_months, constrain_BR_biomes]
    subset_function_args = [{'year_range': year_range},
                            {'months_of_year': months_of_year},
                            {'biome_ID': [biome_ID]}]

    grab_old_trace = True # set to True till you get the code running. 
                          # Then set to False when you start adding in new response curves

    ### output info
    dir_outputs = 'outputs/'

    filename = '-frac_points_' + str(fraction_data_for_sample) + str(len(x_filen_list)) + \
              '-Month_' +  '_'.join([str(mn) for mn in months_of_year])

    
    """ 
        RUN optimization 
    """
    trace, scalers, variable_info_file = \
                     train_MaxEnt_model(y_filen, x_filen_list, CA_filen, priors, 
                                        dir_training, 
                                        filename, dir_outputs,
                                        fraction_data_for_sample,
                                        subset_function, subset_function_args,
                                        niterations, cores, model_title, '',  grab_old_trace,
                                        min_data_points_for_sample = min_data_points_for_sample)
    

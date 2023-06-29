#import multiprocessing as mp 
#mp.set_start_method('forkserver')
import sys
sys.path.append('fire_model/')
sys.path.append('libs/')

from MaxEntFire import MaxEntFire

from read_variable_from_netcdf import *
from plot_maps import *
import os
from   io     import StringIO
import numpy  as np
import math

import pymc  as pm
import pytensor
import pytensor.tensor as tt

import matplotlib.pyplot as plt
import re

import sys
import arviz as az


def MaxEnt_on_prob(BA, fx):
    """calculates the log-transformed continuous logit likelihood for x given mu when x 
       and mu are probabilities between 0-1. 
       Works with tensor variables.   
    Arguments:
        x -- x in P(x|mu). tensor 1-d array
	mu -- mu in P(x|mu). tensor 1-d array
    Returns:
        1-d tensor array of liklihoods.
    """
    fx = tt.switch(
        tt.lt(fx, 0.0000000000000000001),
        0.0000000000000000001, fx)
    return BA*tt.log(fx) + (1.0-BA)*tt.log((1-fx))   

def fit_MaxEnt_probs_to_data(Y, X, niterations, 
                             out_dir = 'outputs/', filename = '', grab_old_trace = True,
                             *arg, **kw):
    """ Bayesian inerence routine that fits independant variables, X, to dependant, Y.
        Based on the MaxEnt solution of probabilities. 
    Arguments:
        Y-- dependant variable as numpy 1d array
	X -- numpy 2d array of indepenant variables, each columne a different variable
	niterations -- number of iterations per chain when sampling the postior during 
                NUTS inference 
		(note default chains is normally 2 and is set by *args or **kw)
	out_dir --string of path to output location. This is where the traces netcdf file 
                will be saved.
		Defauls is 'outputs'.
	filename -- string of the start of the traces output name. Detault is blank. 
		Some metadata will be saved in the filename, so even blank will 
                save a file.
	grab_old_trace -- Boolean. If True, and a filename starting with 'filename' and 
                containing some of the same setting (saved in filename) exists,  it will open 
                and return this rather than run a new one. Not all settings are saved for 
                identifiation, so if in doubt, set to 'False'.
	*args, **kw -- arguemts passed to 'pymc.sample'

    Returns:
        pymc traces, returned and saved to [out_dir]/[filneame]-[metadata].nc
    """

    trace_file = out_dir + '/' + filename + '-nvariables_' + '-ncells_' + str(X.shape[0]) + \
                str(X.shape[1]) + '-niterations_' + str(niterations * cores) + '.nc'
    
    ## check if trace file exsts and return if wanted
    if os.path.isfile(trace_file) and grab_old_trace: 
        return az.from_netcdf(trace_file), trace_file

    trace_callback = None
    try:
        if "SLURM_JOB_ID" in os.environ:
            def trace_callback(trace, draw):        
                if len(trace) % 10 == 0:
                    print('chain' + str(draw[0]))
                    print('trace' + str(len(trace)))
    except:
        pass        

    with pm.Model() as max_ent_model:
        ## set priorts
        betas = pm.Normal('betas', mu = 0, sigma = 1, shape = X.shape[1], 
                          initval =np.repeat(0.5, X.shape[1]))

        powers = pm.Normal('powers', mu = 0, sigma = 1, shape = [2, X.shape[1]])
        ## build model
        
        prediction = MaxEntFire(betas, powers, inference = True).fire_model(X)  
        
        
        ## define error measurement
        error = pm.DensityDist("error", prediction, logp = MaxEnt_on_prob, observed = Y)
                
        ## sample model
        
        trace = pm.sample(niterations, return_inferencedata=True, 
                          callback = trace_callback, *arg, **kw)
        ## save trace file
        trace.to_netcdf(trace_file)
    return trace, trace_file

def train_MaxEnt_model(y_filen, x_filen_list, dir = '', filename_out = '',
                       dir_outputs = '',
                       frac_random_sample = 1.0,
                       subset_function = None, subset_function_args = None,
                       niterations = 100, cores = 4, grab_old_trace = False):
    
   
    Y, X, lmask, scalers = read_all_data_from_netcdf(y_filen, x_filen_list, 
                                                     add_1s_columne = True, dir = dir,
                                                     x_normalise01 = True, 
                                                     frac_random_sample = frac_random_sample,
                                                     subset_function = subset_function, 
                                                     subset_function_args = subset_function_args)
    
    trace, trace_file = fit_MaxEnt_probs_to_data(Y, X, out_dir = dir_outputs, 
                                                 filename = filename, 
                                                 niterations = niterations, cores = cores,
                                                 grab_old_trace = grab_old_trace)
    
    az.plot_trace(trace)
    fig_dir = dir_outputs + '/figs/'
    if not os.path.exists(fig_dir): os.makedirs(fig_dir)

    plt.savefig(fig_dir + filename + '-traces.png')
    
    return trace, trace_file, scalers

def predict_MaxEnt_model(trace, y_filen, x_filen_list, scalers, dir = '', 
                         dir_outputs = '', filename_out = '',
                         subset_functionm = None, subset_function_args = None,
                         paramSamples = 1, hyperparamSamples = 1,
                         trace_file = None):

    Y, X, lmask, scalers = read_all_data_from_netcdf(y_filen, x_filen_list, 
                                                     add_1s_columne = True, dir = dir,
                                                     x_normalise01 = True, scalers = scalers,
                                                     subset_function = subset_function, 
                                                     subset_function_args = subset_function_args)
    
    Obs = read_variable_from_netcdf(y_filen, dir,
                                    subset_function = subset_function, 
                                    subset_function_args = subset_function_args)

    def select_post_param(name, i = None): 
        out = trace.posterior[name].values
        A = out.shape[0]
        B = out.shape[1]
        new_shape = ((A * B), *out.shape[2:])
        out = np.reshape(out, new_shape)
        if i is not None: out = out[i,:]
        return out

    def sample_model(i): 
        print("Sampling interation:" + str(i))
        powers = select_post_param('powers', i)
        betas = select_post_param('betas', i)
        model = MaxEntFire(betas, powers)
        burnt_area = model.burnt_area(X = X)
        burnt_area_probs = model.burnt_area_likelihoods(BA = burnt_area, 
                                                        nesembles = hyperparamSamples)
        return burnt_area, burnt_area_probs
    
    def insert_sim_into_cube(data, eg_cube, dimname = 'realization'):
        def addEnsemble_member(i):  
            Pred = eg_cube.copy()
            pred = Pred.data.copy().flatten()
            pred[lmask] = data[i,:]
            Pred.data = pred.reshape(Pred.data.shape)
            
            coord =  iris.coords.AuxCoord(i, dimname)
            try:
                Pred.remove_coord(dimname)
            except:
                pass        
            Pred.add_aux_coord(coord)

            return(Pred)
        
        Preds = [addEnsemble_member(i) for i in range(data.shape[0])]
        
        return iris.cube.CubeList(Preds).merge_cube()

    if trace_file is not None:
        posterior_file = trace_file[:-3] + '-posterior_samples-' + str(paramSamples)
        posterior_file_uncertainty = posterior_file + '-uncertainty.nc'        
        posterior_file_error = posterior_file + \
                               '-hyperSamples-' + str(hyperparamSamples) + '-uncertainty.nc'
    if  trace_file is not None and os.path.isfile(posterior_file_uncertainty) and \
        os.path.isfile(posterior_file_error) and grab_old_trace: 
        Uncertainty = iris.load_cube(posterior_file_uncertainty)
        Error = iris.load_cube(posterior_file_error)
    else:   
        nits = np.prod(trace.posterior['betas'].values.shape[0:2])
        idx = range(0, nits, int(np.floor(nits/paramSamples)))

        ys = list(map(sample_model, idx))
        
        Uncertainty = np.array([y[0] for y in ys])
        Error = np.array([y[1] for y in ys])
        Error = Error.reshape([Error.shape[0] * Error.shape[1], Error.shape[2]])
        
        Uncertainty = insert_sim_into_cube(Uncertainty, Obs)
        Error = insert_sim_into_cube(Error, Obs)

        iris.save(Uncertainty, posterior_file_uncertainty)
        iris.save(Error, posterior_file_error)
    
    def annual_mean_percentile(cube, q = [5, 95]):
        cube = cube.collapsed('time', iris.analysis.MEAN)
        cube = cube.collapsed('realization', iris.analysis.PERCENTILE, percent=q)
        return cube
    
    
    Uncertainty = annual_mean_percentile(Uncertainty)
    Error = annual_mean_percentile(Error)

    def plot_map(cube, plot_name, plot_n):
        plot_annual_mean(cube, levels, cmap, plot_name = plot_name, scale = 100*12, 
                     Nrows = 2, Ncols = 3, plot_n = plot_n)
  
    plot_map(Obs, "Observtations", 1)
    plot_map(Uncertainty[0], "Uncertainty - 10%", 2)
    plot_map(Uncertainty[1], "Uncertainty - 90%", 3)
    plot_map(Error[0], "Error - 10%", 5)
    plot_map(Error[1], "Error - 90%", 6)
    plt.gcf().set_size_inches(8, 6)
    
    fig_dir = dir_outputs + '/figs/'
    if not os.path.exists(fig_dir): os.makedirs(fig_dir)
    plt.savefig(fig_dir + filename_out + '-maps-2.png')


if __name__=="__main__":
    """ Running optimization and basic analysis. 
    Variables that need setting:
    For Optimization:
        dir_training -- The directory of the training data inclduing 
            dependant and independant variables
        y_filen -- filename of dependant variable (i.e burnt area)
        x_filen_list -- filanames of independant variables
            (ie bioclimate, landscape metrics etc)
        cores - how many chains to start (confusiong name, I know).
            When running on slurm, also number of cores
        fraction_data_for_sample -- fraction of gridcells used for optimization
        niterations -- number of iterations or samples )after warm-up) in optimixation for each
            chain. Equilivent to number of ensemble members.
        months_of_year --- which months to extact on training and projecting
        grab_old_trace -- Boolean. If True and there's an appripritate looking old trace file, 
            then  optimisation is skipped that file is loaded instead. 
            This isn't totally infalable, so if doing a final run and in doubt, set to False
    For Projection/evaluating:
        dir_outputs -- where stuff gets outputted
        dir_projecting -- The directory of the data used for prjections. 
            Should contain same files for independant varibales as dir_training 
            (though you should be able to adpated this easily if required). 
            Doesnt need dependant variable, but if there, this will (once
            we've implmented it) attempt some evaluation.
        paramSamples -- how many iterations (samples) from optimization should be used 
            for plotting and evaluation.
        hyperparamSamples -- how many samples using error hyperparameters per paramSamples
            for plotting and evaluation.
        levels -- levels on the colourbar on observtation and prodiction maps
        cmap -- levels on the colourbar on observtation and prodiction maps
    Returns:
        trace file, maps, etc (to be added too)
    """

    """ 
        SETPUT 
    """
    """ optimization """
    dir_training = "../ConFIRE_attribute/isimip3a/driving_data/GSWP3-W5E5-20yrs/Brazil/AllConFire_2000_2009/"
     #dir_training = "/gws/nopw/j04/jules/mbarbosa/driving_and_obs_overlap/AllConFire_2000_2009/"
    
    y_filen = "GFED4.1s_Burned_Fraction.nc"

    x_filen_list=["precip.nc", "lightn.nc", "crop.nc", "humid.nc","vpd.nc", "csoil.nc", 
                  "lightn.nc", "rhumid.nc", "cveg.nc", "pas.nc", "soilM.nc", 
                   "totalVeg.nc", "popDens.nc", "trees.nc"]

    grab_old_trace = True
    cores = 4
    fraction_data_for_sample = 0.1
    niterations = 100

    months_of_year = [7]
    
    """ Projection/evaluating """
    dir_outputs = 'outputs/'

    dir_projecting = "../ConFIRE_attribute/isimip3a/driving_data/GSWP3-W5E5-20yrs/Brazil/AllConFire_2010_2019/"
    #dir_training = "/gws/nopw/j04/jules/mbarbosa/driving_and_obs_overlap/AllConFire_2000_2009/"

    paramSamples = 100
    hyperparamSamples = 10

    levels = [0, 0.1, 1, 2, 5, 10, 20, 50, 100] 
    cmap = 'OrRd'

    
    """ 
        RUN optimization 
    """
    subset_function = sub_year_months
    subset_function_args = {'months_of_year': months_of_year}

    filename = '_'.join([file[:-3] for file in x_filen_list]) + \
              '-frac_points_' + str(fraction_data_for_sample) + \
              '-Month_' +  '_'.join([str(mn) for mn in months_of_year])

    #### Optimize
    trace, trace_file, scalers = train_MaxEnt_model(y_filen, x_filen_list, dir_training, 
                                                    filename, dir_outputs,
                                                    fraction_data_for_sample,
                                                    subset_function, subset_function_args,
                                                    niterations, cores, grab_old_trace)


    """ 
        RUN projection 
    """
    dir = "../ConFIRE_attribute/isimip3a/driving_data/GSWP3-W5E5-20yrs/Brazil/AllConFire_2010_2019/"
    predict_MaxEnt_model(trace, y_filen, x_filen_list, scalers, dir_projecting,
                         dir_outputs, filename,
                         subset_function, subset_function_args,
                         paramSamples, hyperparamSamples, trace_file = trace_file)
    

    '''
    #Run the model with first iteration
    simulation1 = fire_model(trace.posterior['betas'].values[0,0,:], X,10False)10
    #Plot against observations (Y)
    plt.plot(Y, simulation1, '.')
    plt.show()

    #when developing plots, use betas = trace.posterior['betas'].values[0,0,:]
    and run with model with fire_model(betas, X, False)
    '''
    set_trace()

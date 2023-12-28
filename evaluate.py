import sys
sys.path.append('fire_model/')
sys.path.append('libs/')

from MaxEntFire import MaxEntFire

from BayesScatter import *
from response_curves import *
from jackknife import *
#from train import *

from read_variable_from_netcdf import *
from combine_path_and_make_dir import * 
from namelist_functions import *
from pymc_extras import *
from plot_maps import *
from parameter_mapping import *

import os
from   io     import StringIO
import numpy  as np
import pandas as pd
import math
from scipy.special import logit, expit

import matplotlib.pyplot as plt
import matplotlib as mpl
import arviz as az

from scipy.stats import wilcoxon

from pdb import set_trace


def plot_BayesModel_signifcance_maps(Obs, Sim, lmask, plot_n = 1, Nrows = 3, Ncols = 2,
                                     figure_filename = None):
    
    def flatten_to_dim0(cube):           
        x = cube.data.flatten()[lmask]        
        x = x.reshape([cube.shape[0], int(len(x)/cube.shape[0])])
        return x
    
    X = flatten_to_dim0(Obs) 
    pv = flatten_to_dim0(Sim[1])    
        
    Y = [flatten_to_dim0(Sim[0][i]) for i in range(Sim[0].shape[0])]
    Y = np.array(Y)

    ax = plt.subplot(Nrows, Ncols, plot_n)
    Xf = X.flatten()
    pvf = pv.flatten()

    none0 =  (Xf != 0)
    Xf0 = np.log10(Xf[none0])
    pvf0 = 10**pvf[none0]

    plot_id = ax.hist2d(Xf0, pvf0, bins=100, cmap='afmhot_r', norm=mpl.colors.LogNorm())
    plt.gcf().colorbar(plot_id[3], ax=ax)
    at = np.unique(np.round(np.arange(np.min(Xf0), np.max(Xf0))))
    plt.xticks(at, 10**at)
    labels = np.array([0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
    plt.yticks(10**labels, labels)
    #set_trace()
    Sim[1].data.mask[Sim[1].data == 0] = True
    
    plot_BayesModel_maps(Sim[1], [0.0, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0], 'copper', '', None, 
                         Nrows = Nrows, Ncols = Ncols, plot0 = plot_n, collapse_dim = 'time',
                         scale = 1, figure_filename = figure_filename + 'obs_liklihood')
    
    ax = plt.subplot(Nrows, Ncols, plot_n + 3)
    BayesScatter(Obs, Sim[0], lmask,  0.000001, 0.000001, ax)
    
    pos = np.mean(X[np.newaxis, :, :] > Y, axis = 0)
    pos[X == 0] = np.nan
    sameness_test = np.nanmean(pos, axis = 0) == np.nanmin(pos, axis = 0)
    pos[:, sameness_test] = np.nan
    
    _, p_value = wilcoxon(pos - 0.5, axis = 0, nan_policy = 'omit')
    
    apos = np.nanmean(pos, axis = 0)

    mask = lmask.reshape([ X.shape[0], int(lmask.shape[0]/X.shape[0])])[0]
    apos_cube = insert_data_into_cube(apos, Obs[0], mask)
    p_value_cube = insert_data_into_cube(p_value, Obs[0], mask)

    plot_annual_mean(apos_cube,[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
                     'RdYlBu_r',  plot_name = "mean bias", 
                     Nrows = Nrows, Ncols = Ncols, plot_n = plot_n + 4,
                     figure_filename = figure_filename + 'obs_post-Position.nc')

    plot_annual_mean(p_value_cube, np.array([0, 0.01, 0.05, 0.1, 0.5, 1.0]), 'copper',   
                     plot_name = "mean bias p-value", 
                     Nrows = Nrows, Ncols = Ncols, plot_n = plot_n + 5,
                     figure_filename = figure_filename + 'obs_post-Pvalue.nc')
    



def compare_to_obs_maps(filename_out, dir_outputs, Obs, Sim, lmask, levels, cmap,
                        dlevels = None, dcmap = None,
                        *args, **kw):    
    
    fig_dir = combine_path_and_make_dir(dir_outputs, '/figs/')
    figure_filename = fig_dir + filename_out + '-evaluation'
    figure_dir =  combine_path_and_make_dir(figure_filename)
    
    plot_BayesModel_maps(Sim[0], levels, cmap, '', Obs, Nrows = 3, Ncols = 3,
                         figure_filename = figure_dir)
    plot_BayesModel_signifcance_maps(Obs, Sim, lmask, plot_n = 4, Nrows = 3, Ncols = 3,
                                     figure_filename = figure_dir)
    
    plt.gcf().set_size_inches(12, 12)
    plt.gcf().tight_layout()
    plt.savefig(figure_filename + '.png')


def evaluate_MaxEnt_model_from_namelist(training_namelist = None, evaluate_namelist = None, 
                                        **kwargs):

    variables = read_variable_from_namelist_with_overwite(training_namelist, **kwargs)
    
    return evaluate_MaxEnt_model(**variables)

    

def evaluate_MaxEnt_model(trace_file, y_filen, x_filen_list, scale_file, CA_filen = None, 
                         dir = '', 
                         dir_outputs = '', model_title = '', filename_out = '',
                         subset_function = None, subset_function_args = None,
                         sample_for_plot = 1, grab_old_trace = False, 
                         response_grouping = None,
                         *args, **kw):

    """ Runs prediction and evalutation of the sampled model based on previously run trace.
    Arguments:
        trace - pymc traces nc or nc fileiles, probably from a 'train_MaxEnt_model' run
	y_filen -- filename of dependant variable (i.e burnt area)
        x_filen_list -- filanames of independant variables
	scalers -- the scalers used during the generation of the trace file that scales 
            x data between 0 and 1 (not that it might not scale the opened variables 
            here between 0 and 1 as different data maybe selected for evaluation). 
            can be csv filename.
        dir -- dir y_filen and  x_filen_list are stored in. default it current dir
        dir_outputs -- directiory where all outputs are stored. String
        model_title - title of model run. A str default to 'no_name'. Used to initially to name 
                the dir everythings stored in.
        filename_out -- string of the start of the traces output name. Detault is blank. 
		Some metadata will be saved in the filename, so even blank will 
                save a file.
        subset_function -- a list of constrain function useful for constraining and resticting 
                data to spatial locations and time periods/months. Default is not to 
                constrain (i.e "None" for no functions")
        subset_function_args -- list of arguements that feed into subset_function
        sample_for_plot -- fraction of gridcells used for optimization
        grab_old_trace -- Boolean. If True, and a filename starting with 'filename' and 
                containing some of the same setting (saved in filename) exists,  it will open 
                and return this rather than run new samples. Not all settings are saved for 
                identifiation, so if in doubt, set to 'False'.
	*args, **kw -- arguemts passed to 'evaluate_model' and 'project_model'
    
    Returns:
        look in dir_outputs + model_title, and you'll see figure and tables from evaluation, 
        projection, reponse curves, jackknifes etc (not all implmenented yet)
    """
    
    dir_outputs = combine_path_and_make_dir(dir_outputs, model_title)
    dir_samples = combine_path_and_make_dir(dir_outputs, '/samples/')     
    dir_samples = combine_path_and_make_dir(dir_samples, filename_out)

    fig_dir = combine_path_and_make_dir(dir_outputs, '/figs/')
    trace = az.from_netcdf(trace_file)
    scalers = pd.read_csv(scale_file).values  

     
    common_args = {
        'y_filename': y_filen,
        'x_filename_list': x_filen_list,
        'dir': dir,
        'scalers': scalers,
        'x_normalise01': True,
        'subset_function': subset_function,
        'subset_function_args': subset_function_args
    }

    if CA_filen is not None:
        Y, X, CA, lmask, scalers = read_all_data_from_netcdf(CA_filename = CA_filen, **common_args)   
    else:
        Y, X, lmask, scalers = read_all_data_from_netcdf(**common_args)
    
    Obs = read_variable_from_netcdf(y_filen, dir,
                                    subset_function = subset_function, 
                                    subset_function_args = subset_function_args)

    
    #plot_basic_parameter_info(trace, fig_dir)
    #paramter_map(trace, x_filen_list, fig_dir) 
    
    common_args = {
        'trace': trace,
        'sample_for_plot': sample_for_plot,
        'X': X,
        'eg_cube': Obs,
        'lmask': lmask,
        'dir_samples': dir_samples,
        'grab_old_trace': grab_old_trace}
    
    Sim = runSim_MaxEntFire(**common_args, run_name = "control", test_eg_cube = True)
    
    common_args['Sim'] = Sim[0]
    #jackknife(x_filen_list, fig_dir = fig_dir, **common_args)

    compare_to_obs_maps(filename_out, dir_outputs, Obs, Sim, lmask, *args, **kw)
    Bayes_benchmark(filename_out, fig_dir, Sim, Obs, lmask)
    for ct in ["initial", "standard", "potential", "sensitivity"]:
        response_curve(curve_type = ct, x_filen_list = x_filen_list, 
                       fig_dir = fig_dir, scalers =  scalers,
                       *args, **kw, **common_args)

    
    

if __name__=="__main__":
    """ Running optimization and basic analysis. 
    Variables that need setting:
    For Optimization:
        model_title -- name of model run. Used as directory and filename.
        trace_file -- netcdf filename containing trace (produced in pymc_MaxEnt_train.py)
        y_filen -- filename of dependant variable (i.e burnt area)
        x_filen_list -- filanames of independant variables
            (ie bioclimate, landscape metrics etc)
        months_of_year --- which months to extact on training and projecting
        dir_outputs -- where stuff gets outputted
        dir_projecting -- The directory of the data used for prjections. 
            Should contain same files for independant varibales as dir_training 
            (though you should be able to adpated this easily if required). 
            Doesnt need dependant variable, but if there, this will (once
            we've implmented it) attempt some evaluation.
        sample_for_plot -- how many iterations (samples) from optimixation should be used 
            for plotting and evaluation.
        levels -- levels on the colourbar on observtation and prodiction maps
        cmap -- levels on the colourbar on observtation and prodiction maps
    Returns:
         (to be added too)
    """

    """ 
        SETPUT 
    """
    ### input data paths and filenames

    sample_for_plot = 100
    levels = [0, 0.1, 1, 2, 5, 10, 20, 50, 100] 
    dlevels = [-20, -10, -5, -2, -1, -0.1, 0.1, 1, 2, 5, 10, 20]
    cmap = 'OrRd'
    dcmap = 'RdBu_r'
    dir_projecting = "../ConFIRE_attribute/isimip3a/driving_data/GSWP3-W5E5-20yrs/Brazil/AllConFire_2000_2009/"
    
    training_namelist = "outputs//train_from_bottom-biome-all-controls-4-pca-pm1-ConFire-noq-forced-lin-spread///variables_info--frac_points_0.00516-Month_7-nvariables_-frac_random_sample0.005-nvars_16-niterations_200.txt"
    """ 
        RUN evaluation 
    """
    evaluate_MaxEnt_model_from_namelist(training_namelist, dir = dir_projecting,
                                        grab_old_trace = True,
                                        sample_for_plot = sample_for_plot,
                                        levels = levels, cmap = cmap,
                                        dlevels = dlevels, dcmap = dcmap,
                                        response_grouping = None)
    
    

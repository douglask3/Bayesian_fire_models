import sys
sys.path.append('fire_model/')
sys.path.append('libs/')

from FLAME import FLAME
from ConFire import ConFire

from BayesScatter import *
from response_curves import *
from jackknife import *


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
from scipy.optimize import linear_sum_assignment

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
    
    none0 =  (Xf != 0) #& (pvf > 0.01)
    Xf0 = np.log10(Xf[none0])
    pvf0 = pvf[none0]#10**pvf[none0]
    pvf0[pvf0 > 0.999] = 0.999
    #set_trace()
    #pvf0 = 10**pvf0
    plot_id = ax.hist2d(Xf0, pvf0, bins=100, cmap='afmhot_r', norm=mpl.colors.LogNorm())
    y_min, y_max = plt.ylim()

    # Define the padding (e.g., 10% of the data range)
    padding = 0.02 * (y_max - y_min)

    #  Set new y-axis limits with the padding
    plt.ylim(y_min - padding, y_max + padding)

    #plt.gcf().colorbar(plot_id[3], ax=ax)
    #try:
    #    plt.gcf().colorbar(plot_id[3], ax=ax)
    #except ValueError as e:
    #    print("Error creating colorbar:")
    #    print("plot_id[3] min:", np.nanmin(plot_id[3].get_array()))
    #    print("plot_id[3] max:", np.nanmax(plot_id[3].get_array()))
    #    print(np.min(Xf0))
    #    print(np.max(Xf0))
    #    raise e
    print("Starting colorbar creation...")
    try:
        data = plot_id[3].get_array()
        print(f"Original data shape: {data.shape}")
        print(f"Original data type: {type(data)}")
        
        if isinstance(data, np.ma.MaskedArray):
            print(f"Data is a MaskedArray with {data.mask.sum()} masked values")
            data = data.filled(np.nan)  # Replace masked values with NaN
        
        print(f"Data after mask handling - min: {np.nanmin(data)}, max: {np.nanmax(data)}")
        print(f"Unique values in data: {np.unique(data)}")
        
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        
        if np.isnan(vmin) or np.isnan(vmax):
            raise ValueError(f"Data contains only NaN values")
        
        if vmin == vmax:
            print(f"Constant data detected: all values are {vmin}")
            if vmin == 0:
                vmin, vmax = -1, 1  # Symmetrical range around 0
            else:
                vmin, vmax = vmin * 0.99, vmax * 1.01  # Small range around the constant value
            print(f"Adjusted range for colorbar: vmin = {vmin}, vmax = {vmax}")
        
        print(f"Final vmin: {vmin}, vmax: {vmax}")
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        print("Updating colormap of the plot...")
        plot_id[3].set_norm(norm)
        
        print("Creating colorbar...")
        plt.gcf().colorbar(plot_id[3], ax=ax)
        print("Colorbar created successfully!")
    except Exception as e:
        print("Error creating colorbar:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"plot_id[3] shape: {plot_id[3].get_array().shape}")
        print(f"plot_id[3] size: {plot_id[3].get_array().size}")
        if plot_id[3].get_array().size > 0:
            print(f"plot_id[3] min: {np.nanmin(plot_id[3].get_array())}")
            print(f"plot_id[3] max: {np.nanmax(plot_id[3].get_array())}")
            print(f"plot_id[3] contains inf: {np.isinf(plot_id[3].get_array()).any()}")
            print(f"plot_id[3] contains NaN: {np.isnan(plot_id[3].get_array()).any()}")
            if isinstance(plot_id[3].get_array(), np.ma.MaskedArray):
                print(f"plot_id[3] is a MaskedArray with {plot_id[3].get_array().mask.sum()} masked values")
        
        print("Skipping colorbar due to error")
        
    print("Continuing with the rest of the function...")

    at = np.unique(np.round(np.arange(np.min(Xf0), np.max(Xf0))))
    if len(at) == 0: at = np.array(np.min(Xf0), np.min(Xf0) + 1)
    plt.xticks(at, 10**at)
    #labels = np.array([0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
    #plt.yticks(10**labels, labels)
    
    try:
        Sim[1].data.mask[Sim[1].data == 0] = True
    except:
        pass
    
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
    
    #Sim[0].data = 100 * Sim[0].data
   # Obs.data = Obs.data * 100
    plot_BayesModel_maps(Sim[0], None, cmap, '', Obs, Nrows = 3, Ncols = 3, scale = 100,
                         figure_filename = figure_dir)
    
    plot_BayesModel_signifcance_maps(Obs, Sim, lmask, plot_n = 4, Nrows = 3, Ncols = 3,
                                     figure_filename = figure_dir)
    
    plt.gcf().set_size_inches(14, 12)
    plt.gcf().tight_layout()
    plt.savefig(figure_filename + '.png', pad_inches=0.1)


def evaluate_MaxEnt_model_from_namelist(training_namelist = None, evaluate_namelist = None, 
                                        **kwargs):

    variables = read_variable_from_namelist_with_overwite(training_namelist, **kwargs)
    variables.update(read_variable_from_namelist_with_overwite(evaluate_namelist, **kwargs))
     
    return evaluate_MaxEnt_model(**variables)

def plot_limitation_maps(fig_dir, filename_out, **common_args):
    limitations = [runSim_MaxEntFire(**common_args, run_name = "control_controls-" + str(i),  
                                     test_eg_cube = False, out_index = i, 
                                     method = 'burnt_area', return_limitations = True)  \
                   for i in range(4)] 
        
    for i in range(len(limitations)):
        coord = iris.coords.DimCoord(i, "model_level_number")
        limitations[i].add_aux_coord(coord)
    limitations = iris.cube.CubeList(limitations).merge_cube()
    mn = np.mean(limitations.data, axis = tuple([2, 3, 4]))
    std = np.std(limitations.data, axis = tuple([2, 3, 4]))
    limitations = limitations-mn [:, :, None, None, None]
    limitations = limitations/std[:, :, None, None, None]

    def select_limitations(slice_B, slice_A):
        dists = [np.sum(np.abs((slice_A[i] - slice_B).data), axis = tuple([1, 2, 3])) \
                 for i in range(slice_A.shape[0])]
        
        dists = np.array(dists)            
            
        row_ind, col_ind = linear_sum_assignment(dists)
            
        return col_ind   

    # Iterate through each B slice and apply the function
    sorted_indices = []
    for b_index in range(limitations.shape[1]):  # Loop through B dimension
        print(b_index)
        sorted_index = select_limitations(limitations[:, b_index, :], limitations[:, 0, :])
        sorted_indices.append(sorted_index)
    sorted_indices = np.transpose(np.array(sorted_indices))

    sorted_lim = limitations.copy()
    sorted_lim.data = np.take_along_axis(limitations.data, 
                                         sorted_indices[:, :, None,None, None], axis=1)
        
    figName = fig_dir + filename_out + '-limitation_maps'
    for i in range(sorted_lim.shape[0]):
        plot_BayesModel_maps(sorted_lim[i], 
                             [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5], 
                             'PiYG', '', None, 
                             Nrows = 5, Ncols = 2, plot0 = i*2,
                             scale = 1, figure_filename = figName)
            
    plt.gcf().set_size_inches(8, 12)
    plt.gcf().tight_layout()
    plt.savefig(figName + '.png')

def evaluate_MaxEnt_model(trace_file, y_filen, x_filen_list, scale_file, 
                          extra_params = None,
                          other_params_file = None, CA_filen = None, 
                          model_class = FLAME,
                          link_func_class = MaxEnt, hyper = True, sample_error = True,
                          dir = '', 
                          dir_outputs = '', model_title = '', filename_out = '',
                          filename_out_ext = '',
                          control_run_name = "control",
                          subset_function = None, subset_function_args = None,
                          sample_for_plot = 1, grab_old_trace = False, 
                          run_response_curves = False, 
                          response_grouping = None, run_only = False, return_inputs = False,
                          Y = None, X = None, lmask = None, scalers = None, *args, **kw):
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
        filename_out_ext -- string that gets added to filename_out for eval figures
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
    if other_params_file is not None:
        readin_params = read_variables_from_namelist(other_params_file)
        if extra_params is not None:
            readin_params.update(extra_params)
        extra_params = readin_params
            
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
        if Y is  None or X is  None or lmask is  None or scalers is  None:
            Y, X, lmask, scalers = read_all_data_from_netcdf(**common_args)
    
    Obs = read_variable_from_netcdf(y_filen, dir,
                                    subset_function = subset_function, 
                                    subset_function_args = subset_function_args)
    
    Obs.data[~np.reshape(lmask, Obs.shape)] = np.nan
    #plot_basic_parameter_info(trace, fig_dir)
    #paramter_map(trace, x_filen_list, fig_dir) 
    
    common_args = {
        'class_object': model_class,
        'link_func_class': link_func_class,
        'hyper': hyper,
        'sample_error': sample_error,
        'trace': trace,
        'extra_params': extra_params,
        'sample_for_plot': sample_for_plot,
        'X': X,
        'eg_cube': Obs,
        'lmask': lmask,
        'dir_samples': dir_samples,
        'grab_old_trace': grab_old_trace}
    
    Sim = runSim_MaxEntFire(**common_args, run_name = control_run_name, test_eg_cube = True)
    
    if run_only: 
        if return_inputs: 
            return Sim, Y, X, lmask, scalers 
        else:
            return Sim
    #plot_limitation_maps(fig_dir, filename_out, **common_args)
    
    common_args['Sim'] = Sim[0]
    #set_trace()
    #jackknife(x_filen_list, fig_dir = fig_dir, **common_args)       
    filename_out += filename_out_ext 
    compare_to_obs_maps(filename_out, dir_outputs, Obs, Sim, lmask, *args, **kw)
    Bayes_benchmark(filename_out, fig_dir, Sim, Obs, lmask)

    if run_response_curves: 
        for ct in ["initial", "standard", "potential", "sensitivity"]:
            response_curve(curve_type = ct, x_filen_list = x_filen_list,
                           fig_dir = fig_dir, scalers =  scalers, 
                           *args, **kw, **common_args)
         
    if return_inputs: 
        return Sim, Y, X, lmask, scalers 
    else:
        return Sim
    
    
if __name__=="__main__":
    """ Running optimization and basic analysis. 
    Variables that need setting:
    For Optimization:
        ,model_title -- name of model run. Used as directory and filename.
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
        levels -- levels on the colourbar on observtation and prediction maps
        cmap -- levels on the colourbar on observtation and prediction maps
    Returns:
         (to be added too)
    """

    """ 
        SETPUT 
    """
    ### input data paths and filenames


    training_namelist = "outputs//ConFire_example///variables_info-Forest_consec_dry_mean_tas_max_crop_pas_cveg_humid_lightn_popDens_precip_soilM_totalVeg_vpd-frac_points_0.02-nvariables_-frac_random_sample0.02-nvars_13-niterations_100.txt"

    config_namelist = "namelists/ConFire_example.txt"


    """ 
        RUN evaluation 
    """
    Sim = evaluate_MaxEnt_model_from_namelist(training_namelist, config_namelist)
    
    
    

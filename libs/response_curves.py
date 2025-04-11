import numpy as np
import math
from pymc_extras import *
from plot_maps import *

import matplotlib.pyplot as plt

from pdb import set_trace

def non_masked_data(cube):
    if type(cube) is tuple: cube = cube[0]
    out = cube.data[cube.data.mask == False].data
    return out[~np.isnan(out)]


def initial_curve_experiment(Sim, Xi, col_to_keep, name, trace, sample_for_plot, 
                              eg_cube, lmask, *args, **kw):
    Sim1, Sim2 = standard_curve_experiment(Sim, Xi, col_to_keep, name, trace, sample_for_plot, 
                              eg_cube, lmask, *args, **kw)
    X = Xi.copy()
    X[:,:] = 0.0
    Sim1 = runSim_MaxEntFire(trace, sample_for_plot, X, eg_cube, lmask, 
                             "/all_to_0", *args, **kw)
    
    return Sim1, Sim2


def standard_curve_experiment(Sim, Xi, col_to_keep, name, trace, sample_for_plot, 
                              eg_cube, lmask, *args, **kw):
    X = Xi.copy()
    other_cols = np.arange(X.shape[1])  # Create an array of all columns
    other_cols = np.setdiff1d(other_cols, col_to_keep)  # Exclude col_to_keep
    
    X[:, other_cols] = 0.0 
    #set_trace()
    Sim2 = runSim_MaxEntFire(trace, sample_for_plot, X, eg_cube, lmask, 
                             name + "/all_but_to_0", *args, **kw)

    
    return Sim, Sim2

def potential_curve_experiment(Sim, Xi, col_to_go, name, trace, sample_for_plot, 
                              eg_cube, lmask, *args, **kw):
    X = Xi.copy()
    X[:, col_to_go] = 0.0
    #X[:, col_to_go] = np.mean(X[:, col_to_go])
    
    Sim2 = runSim_MaxEntFire(trace, sample_for_plot, X, eg_cube, lmask, 
                             name + "/to_0", *args, **kw)
    return Sim, Sim2


def sensitivity_curve_experiment(Sim, Xi, col, name, trace, sample_for_plot, 
                              eg_cube, lmask, *args, **kw):
    dx = 0.001

    def sensitivity_experiment(i):
        X = Xi.copy()
        X[:, i] -= dx/2.0  # Subtract 0.1 of all values for the current column
    
        Sim1 = runSim_MaxEntFire(trace, sample_for_plot, X, eg_cube, lmask, 
                                 name + "/subtract_" + str(dx/2.0), *args, **kw)
        
        X = Xi.copy() #restore values
        X[:, i] += dx/2.0 #add 0.1 to all values for the current column
     
        Sim2 = runSim_MaxEntFire(trace, sample_for_plot, X, eg_cube, lmask, 
                                 name + "/add_" + str(dx/2.0), *args, **kw)
        if isinstance(col, int): 
            return Sim2, Sim1
        else: 
            return Sim2 - Sim1
    
    if isinstance(col, int):
        out = sensitivity_experiment(col)
    else:
        out = [sensitivity_experiment(i) for i in col]
        if len(out) == 2:
            out = set(out)
        else:
            out = tuple([out[0], out[1:]])
    
    return out
   
    X[:, col] -= dx/2.0  # Subtract 0.1 of all values for the current column
    
    Sim1 = runSim_MaxEntFire(trace, sample_for_plot, X, eg_cube, lmask, 
                             name + "/subtract_" + str(dx/2.0), *args, **kw)
        
    X = Xi.copy() #restore values
    X[:, col] += dx/2.0 #add 0.1 to all values for the current column
     
    Sim2 = runSim_MaxEntFire(trace, sample_for_plot, X, eg_cube, lmask, 
                             name + "/add_" + str(dx/2.0), *args, **kw)

    
    return Sim1, Sim2

                                 

def response_curve(Sim, curve_type, trace, sample_for_plot, X, eg_cube, lmask, 
                   dir_samples, fig_dir, grab_old_trace, extra_params, 
                   class_object, link_func_class,
                   x_filen_list, 
                   levels, cmap, dlevels, dcmap, scalers = None, 
                   response_grouping = None, *args, **kw):  
    
    figure_filename = fig_dir + curve_type + '-response'
    figure_dir = combine_path_and_make_dir(figure_filename + '-maps/') 
    
    print("Plotting response curve: " + curve_type)
    
    map_type = 1
    if curve_type == "initial":
        response_FUN = initial_curve_experiment
        map_type = 0
    elif curve_type == "standard":
        response_FUN = standard_curve_experiment
    elif curve_type == "potential":
        response_FUN = potential_curve_experiment
    elif curve_type == "sensitivity":
        response_FUN = sensitivity_curve_experiment
        map_type = 2
    else:
        set_trace()

    fig_curve = plt.figure()
    fig_map = plt.figure()

    fcol = math.floor(math.sqrt(X.shape[1]))
    frw = math.ceil(X.shape[1] / fcol)
    
    Ncol = 7
    if map_type == 2:
        Ncol = 8
    
    def plotFun(cube, ylab='', plot0=0, lvls=levels, cm=cmap, **kw2): 
        if map_type > -1:
            plot_BayesModel_maps(cube, lvls, cm, ylab=ylab,
                                 Nrows=len(x_filen_list) + 1, Ncols=Ncol, plot0=plot0, 
                                 scale = 100,
                                 colourbar=True, fig=fig_map, **kw2)
    
    plotFun(Sim, 'Control')

    def process_variables(Sim, X, response_FUN, group_index, g_index,  varname, trace, 
                          sample_for_plot, 
                          eg_cube, lmask, dir_samples, grab_old_trace, map_type, plotFun, 
                          figure_dir, x_filen_list, scalers=None):
        
        Sim1, Sim2i = response_FUN(Sim, X, g_index, varname, trace, sample_for_plot, 
                              eg_cube, lmask, dir_samples, grab_old_trace, 
                              extra_params = extra_params, class_object = class_object, 
                              link_func_class = link_func_class)
        Sim2 = Sim2i[0] if isinstance(Sim2i, list) else Sim2i
        plotN = Ncol * (group_index + 1)
        if map_type >= 0:
            plotFun(Sim2, varname, plotN)
            plotNi = 0
            diff = Sim2.copy() - Sim1.data if Sim1 is not None else Sim2
        if map_type == 2:
            
            plotFun(Sim1, '', plotN + 2, figure_filename=figure_dir + varname + '-absolute')
            
            plotNi = 2
            diff = Sim2.copy()
            if type(g_index) == type(1) or len(g_index) == 1:
                diff.data = diff.data - Sim1.data if Sim1 is not None else Sim2
            else:
                diff.data = np.sqrt(Sim1.data**2 + np.sum([i.data**2 for i in Sim2i], axis = 0))
        
        diffP = diff.collapsed('time', iris.analysis.MEAN)
        
         
        plotFun(diffP, '', plotN + 2 + plotNi, dlevels, dcmap, 
                figure_filename=figure_dir + varname + '-difference')  

        
        if map_type >= 0:
            agree = diffP.copy()
            agree.data = agree.data < 0
            agree = agree.collapsed('realization', iris.analysis.MEAN)
            plot_annual_mean(agree, [1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99], 'PuOr', scale = 100, 
                             Nrows=len(x_filen_list) + 1, Ncols=Ncol, 
                             plot_n = plotN + 4+ plotNi + 1,
                             figure_filename=figure_dir + varname + '-agree.nc')

        ax = fig_curve.add_subplot(frw,fcol, group_index + 1)
        variable_name = varname
        ax.set_title(variable_name)

        if isinstance(g_index, int) or len(g_index) == 1:
            num_bins = 100
            hist, bin_edges = np.histogram(X[:, group_index], bins=num_bins)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            median_values = []
            percentile_10 = []
            percentile_90 = []
    
            for i in range(num_bins):
                mask = (X[:, group_index] >= bin_edges[i]) & (X[:, group_index] < bin_edges[i + 1])
                if np.any(mask):
                    values_in_bin = []
    
                    for rw in range(Sim2.shape[0]):
                        values_in_bin.append(non_masked_data(diff[rw])[mask])
                    values_in_bin = np.array(values_in_bin).flatten()    
    
                    median_values.append(np.median(values_in_bin))
                    percentile_10.append(np.percentile(values_in_bin, 10))
                    percentile_90.append(np.percentile(values_in_bin, 90))
                else:
                    median_values.append(np.nan)
                    percentile_10.append(np.nan)
                    percentile_90.append(np.nan)
    
            if scalers is not None:
                bin_centers = bin_centers*(scalers[1, group_index] - scalers[0, group_index]) + scalers[0, group_index]
            ax.plot(bin_centers, median_values, marker='.', label='Median')
            ax.fill_between(bin_centers, percentile_10, percentile_90, alpha=0.3, 
                            label='10th-90th Percentiles')
    
        else:
            y = X[:, g_index[1]]
            x = X[:, g_index[0]]
            z = non_masked_data(diff[0])
            z = z[:,None]
        
            for rw in range(1, Sim2.shape[0]):
                z = np.concatenate((z, non_masked_data(diff[rw])[:, None]), axis = 1)
            z = np.transpose(np.percentile(z, [10, 50, 90], axis=1))

            output_array = np.column_stack((x, y, z))
            
            x = x*(scalers[1, 0] - scalers[0, 0]) + scalers[0, 0]
            y = y*(scalers[1, 1] - scalers[0, 1]) + scalers[0, 1]
            np.savetxt(figure_dir + varname + '-response_surface.csv', 
                             output_array, delimiter=',', 
                             header = "x,y,p10%,p50%,p90%")

            sample_indices = np.random.choice(len(x), size=1000, replace=False)
            x = x[sample_indices]
            y = y[sample_indices]
            z = z[sample_indices, :]
            ax.scatter(x, y, c=z[:,1], cmap='viridis', marker='o', s=20)
            
    if response_grouping is not None:
        for group_index, group in enumerate(response_grouping):
            varname = f"group_{group_index}"
            makeDir(varname)
            
            g_index = [any([j == i for j in group]) for i in x_filen_list]
            g_index = np.where(g_index)[0]
            
            process_variables(Sim, X, response_FUN, group_index,
                              g_index, varname, trace, sample_for_plot, 
                              eg_cube, lmask, dir_samples, grab_old_trace, map_type, plotFun, 
                              figure_dir, x_filen_list, scalers=scalers)
        # Map saving code
        if map_type > -1:
            fig_map.set_size_inches(12, 4*X.shape[1])
            fig_map.tight_layout()
            fig_map.subplots_adjust(left=0.15)
            fig_map.savefig(figure_filename + '-maps.png')
        plt.close(fig_map)

        fig_curve.savefig(figure_filename + '-curves.png')   
        plt.close(fig_curve)
        plt.clf()
    
    else:
        for col in range(X.shape[1]-1):
            varname = x_filen_list[col]
            if varname.endswith(".nc"):
                varname = varname[:-3]
            Sim1, Sim2 = response_FUN(Sim, X, col, varname, trace, sample_for_plot, 
                                    eg_cube, lmask, dir_samples, grab_old_trace, 
                              extra_params = extra_params, class_object = class_object, 
                              link_func_class = link_func_class)
            # Process remaining individual variables
            
            process_variables(Sim, X, response_FUN, col, col, varname, trace, sample_for_plot, 
                            eg_cube, lmask, dir_samples, grab_old_trace, map_type, plotFun, 
                            figure_dir, x_filen_list, scalers=scalers)
        # Map saving code
        if map_type > -1:
            fig_map.set_size_inches(12, 4*X.shape[1])
            fig_map.tight_layout()
            fig_map.subplots_adjust(left=0.15)
            fig_map.savefig(figure_filename + '-maps.png')
        plt.close(fig_map)

        fig_curve.savefig(figure_filename + '-curves.png')   
        plt.close(fig_curve)
        plt.clf()
        


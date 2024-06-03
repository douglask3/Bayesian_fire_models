
from train import *
from evaluate import *


from   io     import StringIO
import numpy  as np
import cftime

import matplotlib.pyplot as plt

def call_eval(training_namelist, namelist,
              control_run_name, extra_params = None, run_only = True, *args, **kw):
    return evaluate_MaxEnt_model_from_namelist(training_namelist, namelist,
                                               run_only = run_only, 
                                               control_run_name = control_run_name,
                                               extra_params = extra_params, *args, **kw)
    
   
def Standard_limitation(training_namelist, namelist,
                        X, X_id, name, *args, **kws):  
    
    #X[:, col_to_go] = np.mean(X[:, col_to_go])
    Xi = X.copy()
    Xi[:,:] = X.mean(axis=0)
    Xi[:, X_id] = X[:,X_id]
    
    
    return call_eval(training_namelist, namelist,
                     name + '/Standard_'+ str(X_id), hyper = False,
                     *args, **kws)

def Potential_limitation(training_namelist, namelist,
                        X, X_id, name, *args, **kws):   
    
    Xi = X.copy()
    Xi[:,X_id] = np.mean(X[:,X_id])

    return call_eval(training_namelist, namelist,
                     name + '/Potential'+ str(X_id), hyper = False,
                     *args, **kws)

def above_percentile_mean(cube, cube_assess = None, percentile = 0.95):
    
    if cube_assess is None: cube_assess = cube
    area_cube = iris.analysis.cartography.area_weights(cube_assess)
    # Sort the cube by fractional burnt values in descending order
    sorted_indices = np.argsort(cube_assess.data.ravel())
    sorted_cube_data = cube_assess.data.ravel()[sorted_indices]
    area_data_np = np.array(area_cube.data)
    sorted_area_data = area_data_np.ravel()[sorted_indices]

    cumulative_area = np.cumsum(sorted_area_data * sorted_cube_data)

    # Determine the total area of the grid cells
    total_area = np.sum(sorted_area_data * sorted_cube_data)

    # Find the index where the cumulative sum exceeds the 95% threshold of the total area
    threshold_index = np.argmax(cumulative_area > (percentile/100.0) * total_area)

    # Use this index to obtain the fractional burnt value 
    # corresponding to the area-weighted 95th percentile threshold
    threshold_value = sorted_cube_data[threshold_index]
    
    # Mask out grid cells below the area-weighted 95th percentile threshold
    
    masked_cube = cube.copy()
    #masked_cube.data = np.ma.masked_less(cube_assess.data, threshold_value)
    masked_cube.data[cube_assess.data < threshold_value] = np.nan
    masked_cube.data.mask[cube_assess.data < threshold_value] = True
    
    # Calculate the area of the grid cells above the threshold
    masked_area_cube = area_cube.copy()
   
    mean_cube = masked_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN, weights=masked_area_cube)
    #set_trace()#0.00125249
    return(mean_cube)
    
def add_lan_lon_bounds(cube):
    try: 
        cube.coord('latitude').guess_bounds()
    except:
        pass
    
    try:
        cube.coord('longitude').guess_bounds()
    except:
        pass
    return(cube)
    
def make_time_series(cube, name, figName, percentile = None, cube_assess = None, *args, **kw):
    if cube_assess is None: cube_assess = cube
    cube = add_lan_lon_bounds(cube)
    cube_assess = add_lan_lon_bounds(cube_assess)
    
    cube.data = np.ma.masked_invalid(cube.data)
    grid_areas = iris.analysis.cartography.area_weights(cube)
     
    if percentile is None:
        area_weighted_mean =  [cube[i].collapsed(['latitude', 'longitude'],
                                                 iris.analysis.MEAN, weights = grid_areas[i]) \
                                   for i in range(cube.shape[0])]
        area_weighted_mean = iris.cube.CubeList(area_weighted_mean).merge_cube()
        out_dir = figName + '/mean/'
    
    else:
        def percentile_for_relization(cube, i):
            out = [above_percentile_mean(cube[i][j], cube_assess[i][j], percentile, *args, **kw) for j in range(cube.shape[1])]
            for i in range(len(out)): out[i].remove_coord('month')
            for i in range(len(out)): out[i].remove_coord('year')
            out = iris.cube.CubeList(out).merge_cube()
            return(out)

        area_weighted_mean = [percentile_for_relization(cube, i) for i in range(cube.shape[0])]
        area_weighted_mean = iris.cube.CubeList(area_weighted_mean).merge_cube()       
        
        out_dir = figName + '/pc-' + str(percentile) + '/'
    
    makeDir(out_dir)
    out_file = out_dir + 'points-' + name + '.csv'
    np.savetxt(out_file, area_weighted_mean.data, delimiter=',')
    
    TS = area_weighted_mean.collapsed('realization', 
                                      iris.analysis.PERCENTILE, percent=[25, 75])
    
    time_coord = TS.coord('time')
    time_datetime = time_coord.units.num2date(time_coord.points)
    time_datetime = cftime.date2num(time_datetime, 'days since 0001-01-01 00:00:00')/365.24
    TS = np.append(time_datetime[:, None], np.transpose(TS.data), axis = 1)
    
    out_file = figName + '/time_series-' + name + '.csv'
    np.savetxt(out_file, TS, delimiter=',', header = "year,p25%,p75%")
    return TS

def make_both_time_series(*args, **kw):
    #make_time_series(*args, **kw)
    #make_time_series(*args, **kw, percentile = 90)
    make_time_series(*args, **kw, percentile = 95) 

def run_experiment(training_namelist, namelist, control_direction, control_names, 
                   output_dir, output_file, 
                   name = '', *args, **kws):
    if "baseline" in name: 
        run_only = False
    else:
        run_only = True
    print("running: " + name)
    name = name + '-'

    temp_file = 'temp/run_ConFire_lock' + (output_dir + output_file + name).replace('/', '_') + '.txt'
    #if os.path.isfile(temp_file): return None

    figName = output_dir + 'figs/' + output_file + '-' + name + 'control_TS'
    makeDir(figName + '/')

    Evaluate, Y, X, lmask, scalers  = call_eval(training_namelist, namelist,
                        name + '/Evaluate', run_only = run_only, return_inputs = True,
                        *args, **kws)

    Control, Y, X, lmask, scalers  = call_eval(training_namelist, namelist,
                        name + '/control', run_only = True, return_inputs = True, 
                        Y = Y, X = X, lmask = lmask, scalers = scalers,
                        sample_error = False,
                        *args, **kws)
    
    #evaluate_TS = make_both_time_series(Evaluate[0], 'Evaluate', figName, 
    #                                    cube_assess = Control[0])
    
    #control_TS = make_both_time_series(Control[0], 'Control', figName, cube_assess = Control[0])
    
    
    for ltype, FUN in zip(['standard', 'potential'],
                          [Standard_limitation, Potential_limitation]):
        limitation = [FUN(training_namelist, namelist, X, i, 
                      name, *args, 
                      Y = Y, lmask = lmask, scalers = scalers, 
                      cube_assess = Control[0], **kws) \
                        for i in range(X.shape[1])]
        #limitation_TS = np.array([make_both_time_series(cube[0], ltype + '-' + name, figName) \
        #                   for cube, name in zip(limitation, control_names)])
        
    open(temp_file, 'a').close() 

def run_ConFire(namelist):   
    
    run_info = read_variables_from_namelist(namelist) 

    regions = run_info['regions']
    subset_function_args = run_info['subset_function_args']
    

    for region in regions:
        model_title = run_info['model_title'].replace('<<region>>', region)
        dir_training = run_info['dir_training'].replace('<<region>>', region)
        
        subset_function_args['months_of_year'] = run_info['region_months'][region]

        trace, scalers, training_namelist = \
                        train_MaxEnt_model_from_namelist(namelist, model_title = model_title,
                                                         dir_training = dir_training,
                                                         subset_function_args = subset_function_args)
        
        params = read_variables_from_namelist(training_namelist)
        output_dir = params['dir_outputs']
        output_file = params['filename_out']
        
        control_direction = read_variables_from_namelist(params['other_params_file'])
        control_direction = control_direction['control_Direction']
        control_names = read_variables_from_namelist(namelist)['control_names']
    
    
        dir_projecting = run_info['dir_projecting'].replace('<<region>>', region)
        
        
        y_filen = run_info['x_filen_list'][0]
    
        run_experiment(training_namelist, namelist, control_direction, control_names,
                                  output_dir, output_file, 'baseline', 
                                  model_title = model_title, 
                                  subset_function_args = subset_function_args)


if __name__=="__main__":
    namelist = 'namelists/isimip-responses-jules.txt'
    #namelist = 'namelists/SOW2023.txt'
    
    run_ConFire(namelist)
    set_trace()
    #experiment_TS = np.array([make_time_series(cube[0], name)  for cube, name in zip(experiment, experiment_names)])
    
    

    
    
    colors = [(0, 1, 0, 0.5), (0, 0, 1, 0.5), (1, 0, 0, 0.5), (0, 0, 0, 0.5)]  # Green, Blue, Red, Black with 50% opacity
    
    fig, ax = plt.subplots()
    
    for i in range(standard_TS.shape[0]):
        ax.fill_between(range(standard_TS.shape[2]), 
                        standard_TS[i, 0, :], standard_TS[i, 1, :], color=colors[i])

    # Set labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Polygons with Color Vision Impaired Color Combination')
    
    

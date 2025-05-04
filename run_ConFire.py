
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
                        controlID, name, control_direction, *args, **kws):   
    control_Directioni = np.array(control_direction.copy())
    control_Directioni[:] = 0.0
    control_Directioni[controlID] = control_direction[controlID]
    
    extra_params = {"control_Direction": control_Directioni}
    
    return call_eval(training_namelist, namelist,
                     name + '/Standard_'+ str(controlID), extra_params, hyper = False,
                     *args, **kws)

def Potential_limitation(training_namelist, namelist,
                        controlID, name, control_direction, *args, **kws):   
    control_Directioni = np.array(control_direction.copy())
    control_Directioni[controlID] = 0.0
    
    extra_params = {"control_Direction": control_Directioni}
    
    return call_eval(training_namelist, namelist,
                     name + '/Potential'+ str(controlID), extra_params, hyper = False,
                     *args, **kws)

def above_percentile_mean(cube, cube_assess = None, percentile = 0.95):
    print("finding " + str(percentile) + " for year" + str(cube.coord('year').points))
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

    # Find the index where the cumulative sum exceeds the percentile threshold of the total area
    threshold_index = np.argmax(cumulative_area > (percentile/100.0) * total_area)

    # Use this index to obtain the fractional burnt value 
    # corresponding to the area-weighted percentile threshold
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
     
    if percentile is None or percentile == 0.0:
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

def make_both_time_series(percentiles, *args, **kw):
    if percentiles is None: return None
    for percentile in percentiles:
        make_time_series(*args, **kw, percentile = percentile) 
    #make_time_series(*args, **kw)
    #make_time_series(*args, **kw, percentile = 90)
    #make_time_series(*args, **kw, percentile = 95) 

def run_experiment(training_namelist, namelist, control_direction, control_names, 
                   output_dir, output_file, 
                   name = '', time_series_percentiles = None, *args, **kws):
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
                        filename_out_ext = 'stochastic',
                        *args, **kws)

    Control, Y, X, lmask, scalers  = call_eval(training_namelist, namelist,
                        name + '/control', run_only = run_only, return_inputs = True, 
                        Y = Y, X = X, lmask = lmask, scalers = scalers,
                        sample_error = False, filename_out_ext = 'none_stochastic',
                        *args, **kws)
    
    
    evaluate_TS = make_both_time_series(time_series_percentiles, Evaluate[0], 'Evaluate', 
                                        figName, cube_assess = Control[0])
    
    control_TS = make_both_time_series(time_series_percentiles, Control[0], 'Control', 
                                       figName, cube_assess = Control[0])

    
    if  control_names is None: return None

    for ltype, FUN in zip(['standard', 'potential'],
                          [Standard_limitation, Potential_limitation]):
        
        limitation = [FUN(training_namelist, namelist, i, 
                      name, control_direction, *args, 
                      Y = Y, X = X, lmask = lmask, scalers = scalers, 
                      cube_assess = Control[0], **kws) \
                        for i in range(len(control_direction))]
        limitation_TS = np.array([make_both_time_series(time_series_percentiles, \
                                                        cube[0], ltype + '-' + name, figName) \
                           for cube, name in zip(limitation, control_names)])
        
    open(temp_file, 'a').close() 

def run_ConFire(namelist):   
    
    run_info = read_variables_from_namelist(namelist) 

    def select_from_info(item):
        try:
            out = run_info[item]
        except:
            out = None
        return out

    control_direction = select_from_info('control_Direction')
    if control_direction is None:
        control_direction = [param['value'] for param in run_info['priors'] \
                             if param['pname'] == 'control_Direction'][-1]
    
    control_names = select_from_info('control_names')
    subset_function_args = select_from_info('subset_function_args')
    subset_function_eval = select_from_info('subset_function_eval')
    subset_function_args_eval = select_from_info('subset_function_args_eval')
    if subset_function_args_eval is None: subset_function_args_eval =subset_function_args 
    regions = select_from_info('regions')
    time_series_percentiles = select_from_info('time_series_percentiles')

    def run_for_regions(region = None):
        
        if region is None:
            region = '<<region>>'
        else:
            def set_region_months(ssa):
                if isinstance(ssa, list):
                    for i in range(len(ssa)):
                        try:
                            ssa['months_of_year'] = run_info['region_months'][region]
                        except:
                            pass
                else:
                    ssa['months_of_year'] = run_info['region_months'][region]
                return ssa
            
            set_region_months(subset_function_args)
            set_region_months(subset_function_args_eval)
        model_title = run_info['model_title'].replace('<<region>>', region)
        dir_training = run_info['dir_training'].replace('<<region>>', region)
        dir_projecting = run_info['dir_projecting'].replace('<<region>>', region)
        
        trace, scalers, training_namelist = \
                        train_MaxEnt_model_from_namelist(namelist, model_title = model_title,
                                                         dir_training = dir_training,
                                                         subset_function_args = subset_function_args)
        params = read_variables_from_namelist(training_namelist)
        output_dir = params['dir_outputs']
        output_file = params['filename_out']

        def find_replace_period_model(exp_list):
            exp_list_all = [item.replace('<<region>>', region) for item in exp_list \
                            if "<<experiment>>" not in item and "<<model>>" not in item]
            looped_items = [item for item in exp_list \
                            if "<<experiment>>" in item and "<<model>>" in item]
            for experiment, period in zip(experiments, periods):
                for model in models:
                    dirs = [item.replace("<<period>>", period) for item in looped_items]    
                    dirs = [item.replace("<<model>>", model) for item in dirs]   
                    dirs = [item.replace("<<experiment>>", experiment) for item in dirs] 
                    dirs = [item.replace('<<region>>', region) for item in dirs] 
                    exp_list_all += dirs
             
            return exp_list_all
    
        

        y_filen = run_info['x_filen_list'][0]
    
        run_experiment(training_namelist, namelist, control_direction, control_names,
                                  output_dir, output_file, 'baseline', 
                                  model_title = model_title, 
                                  subset_function = subset_function_eval,
                                  subset_function_args = subset_function_args_eval)
        
        try:
            experiment_dirs  = run_info['experiment_dir']
            experiment_names = run_info['experiment_names']
            experiments = run_info['experiment_experiment']
            periods = run_info['experiment_period']
            models = run_info['experiment_model']
            experiment_dirs = find_replace_period_model(experiment_dirs)
            experiment_names = find_replace_period_model(experiment_names)

            [run_experiment(training_namelist, namelist, control_direction, 
                                         control_names,
                                         output_dir, output_file, name, dir = dir, 
                                         y_filen = y_filen, model_title = model_title,
                                         subset_function_args = subset_function_args) \
                              for name, dir in zip(experiment_names, experiment_dirs)]
        except:
            pass


    if regions is None:
        run_for_regions(None)
    else:
        for region in regions: run_for_regions(region)



if __name__=="__main__":
    namelist = 'namelists/ConFire_Canada.txt'
    namelist = 'namelists/Greece.txt'
    namelist = 'namelists/tuning.txt'
    namelist = 'namelists/isimip-fwi2.txt'
    namelist = 'namelists/isimip3.txt'
    namelist = 'namelists/isimip2425.txt'
    namelist = 'namelists/isimip2425-test.txt'
    #namelist = 'namelists/ar7_ref.txt'
    #namelist = 'namelists/ar7_simple.txt'
    #namelist = 'namelists/Pantanal_example.txt'
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
    
    

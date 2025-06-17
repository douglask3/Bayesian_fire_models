from train import *
from evaluate import *
from iris_plus import add_bounds

from   io     import StringIO
import numpy  as np
import cftime
import matplotlib.pyplot as plt

import datetime

import sys
sys.path.append('libs/')
from climtatology_difference import *

try:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
except:
    pass

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
    if cube_assess is None: cube_assess = cube
    area_cube = iris.analysis.cartography.area_weights(cube_assess)
    
    # Sort the cube by fractional burnt values in descending order
    sorted_indices = np.argsort(cube_assess.data.ravel())
    sorted_cube_data = cube_assess.data.ravel()[sorted_indices]
    area_data_np = np.array(area_cube.data)
    sorted_area_data = area_data_np.ravel()[sorted_indices]

    cumulative_area = np.cumsum(sorted_area_data * sorted_cube_data)

    # Determine the total area of the grid cells
    total_area = np.nansum(sorted_area_data * sorted_cube_data)

    # Find the index where the cumulative sum exceeds the percentile threshold of the total area
    threshold_index = np.argmax(cumulative_area > (percentile/100.0) * total_area)

    # Use this index to obtain the fractional burnt value 
    # corresponding to the area-weighted percentile threshold
    threshold_value = sorted_cube_data[threshold_index]#

    mask = (cube_assess.data >= threshold_value) & (~cube_assess.data.mask)
    return np.sum(cube.data[mask] * area_data_np[mask]) / np.sum(area_data_np[mask])

  
def make_time_series(cube, name, output_path, percentile = None, cube_assess = None, 
                     grab_old = False, *args, **kw):
    print("finding " + str(percentile) + " for " + name + "\n\t into:" + output_path)
    print(datetime.datetime.now())
    if percentile is None or percentile == 0.0:        
        out_dir = output_path + '/mean/'
    else:
        out_dir = output_path + '/pc-' + str(percentile) + '/'
    
    lock_file = out_dir + '.txt'
    if os.path.isfile(lock_file) and grab_old:    
        return out_dir
    
    if cube_assess is None: cube_assess = cube
    cube = add_bounds(cube)
    cube_assess = add_bounds(cube_assess)
    
    cube.data = np.ma.masked_invalid(cube.data)
    grid_areas = iris.analysis.cartography.area_weights(cube)
            
    if percentile is None or percentile == 0.0:
        area_weighted_mean =  [cube[i].collapsed(['latitude', 'longitude'],
                                                 iris.analysis.MEAN, weights = grid_areas[i]) \
                                   for i in range(cube.shape[0])]
        
        area_weighted_mean = iris.cube.CubeList(area_weighted_mean).merge_cube().data
    
    else:
        def percentile_for_relization(cube, i):
            print("\tprocessing enemble" + str(i))
            out = [above_percentile_mean(cube[i][j], cube_assess[i][j], percentile, *args, **kw) for j in range(cube.shape[1])]
            return out

        area_weighted_mean = np.array([percentile_for_relization(cube, i) \
                                      for i in range(cube.shape[0])])      
    
    climatology, anomaly, ratio = climtatology_difference(area_weighted_mean)
    makeDir(out_dir)
    def output_cube_to_csv(data, realizations, extra_dim,  filename): 
        times = cube.coord('time').units.num2date(cube.coord('time').points)[0:data.shape[1]]
        df = pd.DataFrame(data, index=realizations, columns=[t.isoformat() for t in times])
        df.index.name = extra_dim
        df.to_csv(filename)
        #np.savetxt(out_file_points, area_weighted_mean.data, delimiter=',')
    

    percentiles = [5, 10, 25, 50, 75, 90, 95]

    def make_output_TS(data, dir = ''):
        
        out_file_points = out_dir + '/members/' + dir + '/'
        out_file_TS = out_dir + '/percentles/' + dir + '/' 
        makeDir(out_file_points)
        makeDir(out_file_TS)
        output_cube_to_csv(data, cube.coord('realization').points, 
                       'realization', out_file_points  + name + '.csv')
        TS = np.nanpercentile(data, percentiles, axis = 0)
        output_cube_to_csv(TS, percentiles,  'percentiles', out_file_TS  + name + '.csv')
    
    make_output_TS(area_weighted_mean, 'absolute')
    make_output_TS(climatology, 'climatology')
    make_output_TS(anomaly, 'anomaly')
    make_output_TS(ratio, 'ratio')
    
    return out_dir

def make_both_time_series(percentiles, *args, **kw):
    if percentiles is None: return None
    for percentile in percentiles:
        make_time_series(*args, **kw, percentile = percentile) 


def run_experiment(training_namelist, namelist, control_direction, control_names, 
                   output_dir, output_file, 
                   name = '', time_series_percentiles = None, 
                   limitation_types = None, controls_to_plot = None,*args, **kws):
    
    if "baseline" in name: 
        run_only = False
    else:
        run_only = True
    print("running: " + name)
    name = name + '-'
    
    temp_file = 'temp/run_ConFire_lock' + (output_dir + \
            output_file + name).replace('/', '_') + '.txt'
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
    
    grab_old = read_variables_from_namelist(namelist)['grab_old_trace']
    out_dir_ts = output_dir +'/time_series/' +  output_file + '/' + name
    
    evaluate_TS = make_both_time_series(time_series_percentiles, Evaluate[0], 'Evaluate', 
                                        out_dir_ts,
                                        cube_assess = Control[0], grab_old = grab_old)
    
    control_TS = make_both_time_series(time_series_percentiles, Control[0], 'Control', 
                                       out_dir_ts,
                                       cube_assess = Control[0], grab_old = grab_old)
    
    if limitation_types is not None and controls_to_plot is not None:
        if control_names is None: 
            control_names = [srt(i) for i in controls_to_plot]
        limitation_types_funs = []
        if 'standard' in limitation_types:
            limitation_types_funs += [Standard_limitation]
        if 'potential' in limitation_types:
            limitation_types_funs += [Potential_limitation]
    
        for ltype, FUN in zip(limitation_types,limitation_types_funs):
            limitation = [FUN(training_namelist, namelist, i, 
                          name, control_direction, *args, 
                          Y = Y, X = X, lmask = lmask, scalers = scalers, 
                              cube_assess = Control[0], **kws) \
                        for i in controls_to_plot]
            limitation_TS = np.array([make_both_time_series(time_series_percentiles, \
                                                        cube[0], \
                                                        ltype + '-' + name, out_dir_ts, \
                                                        grab_old = grab_old) \
                               for cube, name in zip(limitation, control_names)])
        
    open(temp_file, 'a').close() 

def run_experiment_wrapper(kwargs):
    try:
        run_experiment(**kwargs)
        return (kwargs, "success")
    except Exception as e:
        print(f"[ERROR] Experiment failed with kwargs={kwargs['name']}:\n{e}")
        import traceback
        traceback.print_exc()
        return (kwargs, f"error: {e}")

def run_ConFire(namelist):   
    print(f"Running ConFLAME with namelist: {namelist}")
    run_info = read_variables_from_namelist(namelist) 

    def select_from_info(item, alternative = None):
        try:
            out = run_info[item]
        except:
            out = alternative
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
            if select_from_info('region_mnths') is not None:
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
            if periods is None and periods is None: 
                return exp_list_all
            for experiment, period in zip(experiments, periods):
                for model in models:
                    dirs = [item.replace("<<period>>", period) for item in looped_items]    
                    dirs = [item.replace("<<model>>", model) for item in dirs]   
                    dirs = [item.replace("<<experiment>>", experiment) for item in dirs] 
                    dirs = [item.replace('<<region>>', region) for item in dirs] 
                    exp_list_all += dirs
             
            return exp_list_all
        
        y_filen = [run_info['y_filen']]
        names_all = ['baseline']
        exp_type = ['single']        
        dirs_all = [params['dir']]
        
        try:
            y_filen1 = [select_from_info('y_filen_eval', run_info['x_filen_list'][0])]
            experiment_dirs  = select_from_info('experiment_dir')
            experiment_names = select_from_info('experiment_names')
            experiments = select_from_info('experiment_experiment')
            periods = select_from_info('experiment_period')
            models = select_from_info('experiment_model')
            limitation_types = select_from_info('limitation_types')
            controls_to_plot = select_from_info('controls_to_plot', 
                                                 range(len(control_direction)))
            experiment_dirs = find_replace_period_model(experiment_dirs)
            experiment_names = find_replace_period_model(experiment_names)
            exp_type = exp_type + \
                select_from_info('experiment_type', ['single'] * len(experiment_names))
            names_all = names_all + experiment_names
            dirs_all = dirs_all + experiment_dirs
            y_filen = y_filen + y_filen1 * len(experiment_dirs)
        except:
            pass   
         
        args_list = [dict(training_namelist=training_namelist,
                          namelist=namelist,
                          control_direction=control_direction,
                          control_names=control_names,
                          output_dir=output_dir,
                          output_file=output_file,
                          name=name,
                          time_series_percentiles=time_series_percentiles,
                          limitation_types = limitation_types, 
                          controls_to_plot = controls_to_plot,
                          dir=dir,
                          experiment_type = expt,
                          y_filen=yfile,
                          model_title=model_title,
                          subset_function = subset_function_eval,
                          subset_function_args = subset_function_args_eval
                         )
                    for name, dir, expt, yfile in zip(names_all, dirs_all, exp_type, y_filen)
                ]
        args_list.reverse()
        if len(args_list) > 1 and select_from_info('parallelize', True): 
            try:
                with get_context("spawn").Pool(processes=4) as pool:
                    pool.map(run_experiment_wrapper, args_list)
            except:
                for args in args_list:
                    run_experiment_wrapper(args)
        else:
            for args in args_list:
                run_experiment_wrapper(args)

    if regions is None:
        run_for_regions(None)
    else:
        for region in regions: run_for_regions(region)

if __name__=="__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python run_ConFire.py <namelist_path>")
        sys.exit(1)
    namelist = sys.argv[1]

    run_ConFire(namelist)
    

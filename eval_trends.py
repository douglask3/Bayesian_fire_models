from main import *
from libs.plot_AR6_hexagons import *
from libs.NME import *
from libs.flatten_list import *
from libs.time_series_comparison import *
from libs.combine_path_and_make_dir import *
import numpy  as np
import matplotlib.pyplot as plt
from pdb import set_trace
import os
import pickle
import glob
    
def trend_prob_for_region(region_code, value, region_type, 
                          filenames_observation, filename_model, 
                          mod_scale, obs_scale, year_range, 
                          n_itertations, traces_dir, log_transform = False, *arg, **kw):# in ar6_regions: 
    
    if region_type is None or region_type == 'ar6':
        if not isinstance(region_code, str): return 
        if region_code == 'EAN' or region_code == 'WAN': return
        if len(region_code) > 5: return
        region_constrain_function = ar6_region
    elif region_type == 'gfed':        
        region_constrain_function = constrain_GFED
    
    print(region_code)
    
    subset_functions = [sub_year_range, region_constrain_function, make_time_series]
    subset_function_args = [{'year_range': year_range},
                        {'region_code' : [region_code]}, 
                        {'annual_aggregate' : iris.analysis.SUM}]
    
    traces_dir = combine_path_and_make_dir(traces_dir, 'REGION-' + region_code)
    trace_file_save = traces_dir + \
                     '_'.join([str(year) for year in year_range]) + \
                     '-model_scale_' +  str(mod_scale) + \
                     '-obs_scale_' + '_'.join([str(i) for i in obs_scale])
    
    Y_temp_file = trace_file_save + '-Y' + '.npy'
    X_temp_file = trace_file_save + '-X' + '.npy'
    
    if os.path.isfile(Y_temp_file) and os.path.isfile(X_temp_file): 
        Y = np.load(Y_temp_file)
        X = np.load(X_temp_file)
    else :
        Y, X, = read_all_data_from_netcdf(filename_model, filenames_observation, 
                                         time_series = year_range, check_mask = False,
                                         subset_function = subset_functions, 
                                         subset_function_args = subset_function_args,
                                         *arg, **kw)
        Y = Y * mod_scale
        X = X * obs_scale
        
        def save_arrany(x, file):
            if np.ma.isMaskedArray(x): x = x.data
            np.save(file, x)  
            return(x)
        
        Y = save_arrany(Y, Y_temp_file)
        X = save_arrany(X, X_temp_file)
        
    gradient_compare = find_and_compare_gradients(Y, X, trace_file_save, 
                                                  log_transform = log_transform, 
                                                  n_itertations = n_itertations)
    
    nme = NME(X, Y)        
    nme_null = NME_null(X)
    obs_mean = np.nanmean(X, axis = 0)
    
    Yspread = np.reshape(np.repeat(Y, X.shape[1]), X.shape)
    Yspread[np.isnan(X)] = np.nan
    mod_mean = np.nanmean(Yspread, axis = 0)
        
        
    out =  region_code, value, obs_mean, mod_mean, nme
    
    out = [out[0], out[1]] + list(np.concatenate(out[2:4])) + list(out[4].values.flatten()) + \
          list(nme_null.values.flatten()) + \
          list(gradient_compare.values.flatten())
    
    return(out)

def eval_trends_over_regions(filenames_observation, observations_names = None,
                             model_name = '', output_table = 'table/', grab_output = False, 
                             region_type = 'ar6',
                             *args, **kw):
    output_file = output_table + model_name + '-results.csv'
    if grab_output and os.path.isfile(output_file) and False: 
        return pd.read_csv(output_file, index_col = 0)
    
    if region_type is None or region_type == 'ar6':
        region_codes =  regionmask.defined_regions.ar6.land.region_ids.items()
    elif region_type == 'gfed':
        region_codes = gfed_region_codes
    else:
        sys.exit("ERROR: region type not defined")
        
    if observations_names is None:
        observations_names = [str(i) for i in range(len(filenames_observation))] + ['All']
    
    NME_obs = observations_names + ['All']
        
    null_models = ['Median', 'Mean', 'Randomly-resampled mean', 'Randomly-resampled - sd']
    index = ['Region Code', 'Region ID'] + \
            ['observation ' + str(i) for i in observations_names] + \
            ['simulations ' + str(i) for i in observations_names] + \
            [['NME ' + j + ' obs. ' + i for i in NME_obs] for j in ['1', '2', '3', 'A']]  + \
            [[j + 'Null model obs. ' + i for i in NME_obs] for j in null_models] + \
            ['Gradient overlap', 'Obs trend - 10%', 'Obs trend - 90%', 
                                  'Mod trend - 10%', 'Mod trend - 90%']
    index = flatten_list(index)
    
    result = list(map(lambda item: trend_prob_for_region(item[0], item[1], region_type, \
                                                         filenames_observation, *args, **kw), \
                      region_codes))

    result = list(filter(lambda x: x is not None, result))
    
    if region_type == 'gfed':
        colOrder = [np.where(np.array(result)[:,0] == rc)[0][0] for rc in gfed_region_order]
        result = [result[i] for i in colOrder]
    
    result = pd.DataFrame(np.array(result).T, index = index, columns = np.array(result)[:,0])
    result.to_csv(output_file)
    
    return result

def NME_by_obs(obs_name, result, *arg, **kw):
    if obs_name == 'All':
        X = result[result.index.str.contains('observation')].values.T.astype(float)
        Y = result[result.index.str.contains('simulation')].values.T.astype(float)
        Y = np.mean(Y, axis = 1)
    else:
        X = result.loc['observation ' + obs_name].values.astype(float)
        Y = result.loc['simulations ' + obs_name].values.astype(float)
    
    nme = NME(X, Y, *arg, **kw)        
    nme_null = NME_null(X, *arg, **kw)
    
    return pd.DataFrame(np.append(nme_null, nme), index = np.append(nme_null.index, nme.index))
    

def run_all_eval_for_model(filename_model, name_model, variable_model,
                  filenames_observation, observations_names,
                  year_range, 
                  n_itertations, tracesID, mod_scale,  obs_scale, units,
                  output_dir, filename_model_exclude = 'rcp2p6',
                  region_type = None, *args, **kw):
    
    combine_path_and_make_dir(output_dir)
    output_dir   = combine_path_and_make_dir(output_dir, region_type)
    output_table = combine_path_and_make_dir(output_dir, '/table/')
    output_figs  = combine_path_and_make_dir(output_dir, '/figs/' )
    output_ncdfs = combine_path_and_make_dir(output_dir, '/gen_data/')
    
    
    traces_dir = output_ncdfs + name_model + '-gradient_trace'

    if '*' in filename_model:
        filename_model = sorted(glob.glob(filename_model))
        if filename_model_exclude is not None:
            filename_model = [file for file in filename_model \
                               if filename_model_exclude not in file]
        def file_with_year(i):
            return np.where([str(year_range[i]) in file for file in filename_model])[0][0]
        id0 = file_with_year(0)
        id1 = file_with_year(1)
        filename_model = filename_model[id0:id1]
    
    result = eval_trends_over_regions(filenames_observation, observations_names, 
                                      name_model, output_table, True,
                                      region_type, filename_model, 
                                      mod_scale, obs_scale,
                                      year_range, n_itertations, traces_dir, 
                                      y_variable = variable_model, *args, **kw)
    
    subset_functions = [sub_year_range, annual_average]
    subset_function_args = [{'year_range': year_range},
                            {'annual_aggregate' : iris.analysis.SUM}]
    
    def annaul_averge_from_map(cube): return np.mean(make_time_series(cube).data)
    def open_compare_obs_mod(filename_obs, scale, name_obs, output_maps, openOnly = False):
        combine_path_and_make_dir(output_maps)
        output_maps = output_maps + '/' + name_obs + '/'
        combine_path_and_make_dir(output_maps)
        print(name_obs)
        def readFUN(filename, subset_function_args, *args, **kw):
            return read_variable_from_netcdf(filename, subset_function = subset_functions, 
                                             make_flat = False, 
                                             subset_function_args = subset_function_args,
                                             *args, **kw)
        

        X, year_range = readFUN(filename_obs, subset_function_args)
        subset_function_args[0]['year_range'] = year_range
        Y, nn = readFUN(filename_model, subset_function_args, variable = variable_model)
        if mod_scale is not None: Y.data = Y.data * mod_scale
        if scale is not None: X.data = X.data * scale
        if units is not None: Y.units = units
        if units is not None: X.units = units
        
        if openOnly: return X, Y
        X_filename = output_maps + 'observation.nc'
        Y_filename = output_maps + 'simulation.nc'
        iris.save(X, X_filename)
        iris.save(Y, Y_filename)
        
        nme = NME_cube(X, Y)
        nme_null = NME_null_cube(X)
        scores = pd.DataFrame(np.append(nme_null, nme), 
                              index = np.append(nme_null.index, nme.index))
        
        return scores, X, Y, year_range, annaul_averge_from_map(X), annaul_averge_from_map(Y)
    
    temp_file_path = 'temp/eval_trends_nme_over_obs_pickle-' + tracesID + '.pkl'
    
    try:
        with open(temp_file_path, "rb") as file:
            X, Y, nme_over_obs, nme_cube_all, nme_null_cube_all = pickle.load(file)
    except:
        nme_over_obs = list(map(lambda x, y, z: open_compare_obs_mod(x, y, z, output_maps), 
                                filenames_observation, obs_scale, observations_names))
         
        #with open(temp_file_path, "wb") as file:
        #    pickle.dump(nme_over_obs, file)
    
        year_range = np.array([out[3] for out in nme_over_obs])
        year_range = [np.min(year_range[:,0]), np.max(year_range[:,1])]
        subset_function_args[0]['year_range'] = year_range

        XYs = list(map(lambda x, y, z: open_compare_obs_mod(x, y, z, 
                                                            output_maps, openOnly = True), 
                                    filenames_observation, obs_scale, observations_names))
    
        X = [xy[0] for xy in XYs]
        cubes = []    
        for i in range(len(X)):       
            coord = iris.coords.DimCoord(i, "realization")  
            cube = X[0].copy()      
            cube.add_aux_coord(coord)
            cube.data = X[i].data.astype(np.float32)
            cubes.append(cube)
        
        X = iris.cube.CubeList(cubes).merge_cube()    
        Y = XYs[0][1]
        output_maps = output_maps + '/All/'
        combine_path_and_make_dir(output_maps)
    
        X_filename = output_maps + 'observation.nc'
        Y_filename = output_maps + 'simulation.nc'
        iris.save(X, X_filename)
        iris.save(Y, Y_filename)
        nme_cube_all = NME_cube(X, Y, x_range = True)
        nme_null_cube_all = NME_null_cube(X, x_range = True)
        with open(temp_file_path, "wb") as file:
            pickle.dump((X, Y, nme_over_obs, nme_cube_all, nme_null_cube_all), file)

    
    def select_i_from_out(i):  return np.array([out[i] for out in nme_over_obs])

    nme_by_cell_obs = select_i_from_out(0)[:,:,0]
    glob_tot_obs = select_i_from_out(4)[:, np.newaxis]
    glob_tot_mod = select_i_from_out(5)[:, np.newaxis]
    nme_by_cell_obs = np.concatenate((glob_tot_obs, glob_tot_mod, nme_by_cell_obs), axis=1)

    
    nme_by_cell_all =  np.concatenate((np.array([annaul_averge_from_map(X)]), 
                                       np.array([annaul_averge_from_map(Y)]), 
                                       nme_null_cube_all.values.flatten(),
                                       nme_cube_all.values.flatten() ))
    nme_by_cell = np.vstack((nme_by_cell_obs, nme_by_cell_all))
     
    nme_by_reg_obs = list(map(lambda obs: NME_by_obs(obs, result), observations_names))
    nme_by_reg_obs = np.array(nme_by_reg_obs)[:,:,0]
    nme_by_reg_all = np.array(NME_by_obs('All', result, x_range = True))[:,0]
    nme_by_reg = np.vstack((nme_by_reg_obs, nme_by_reg_all))
    nme_by_reg = np.hstack((nme_by_cell[:,0:2], nme_by_reg))

    nme_by_both = np.vstack((nme_by_cell, nme_by_reg))
    index = ['Obs total', 'Mod total', 'Median Null', 'Mean Null', 
             'Randomly Resampled Null - mean', 'Randomly Resampled Null - sdev',
             'NME step 1', 'NME step 2', 'NME step 2', 'NME step A']
    
    
    cnames = flatten_list([[obs + ' over cell'] for obs in observations_names + ['All']] + \
                          [[obs + ' over region'] for obs in observations_names + ['All']])   
        
    nme_by_both = pd.DataFrame(nme_by_both.T, index = index, columns = cnames)
    nme_by_both.to_csv(output_file + '-global.csv')
    
    #plot_AR6_hexagons(result, resultID = 41, colorbar_label = 'Gradient Overlap')

def run_all_eval_for_models(filenames_model, names_model, *args, **kw):
    [run_all_eval_for_model(filename_model, name_model, *args, **kw) \
        for filename_model, name_model in zip(filenames_model, names_model)]

if __name__=="__main__":    
    filenames_model = ["/scratch/hadea/isimip3a/u-cc669_isimip3a_fire/20CRv3-ERA5_obsclim/jules-vn6p3_20crv3-era5_obsclim_histsoc_default_burntarea-total_global_monthly_1901_2021.nc",
                      "/hpc/data/d05/cburton/jules_output/u-cf137/GFDL-ESM2M/*.ilamb.*.nc"]
    names_model = ['isimip3a-era5-obsclim', "isimip2b-GFDL-ESM2M"]
    variable_model = 'burnt_area_gb'
    dir_observation = "/data/dynamic/dkelley/fireMIPbenchmarking/data/benchmarkData/"
    filenames_observation = ["ISIMIP3a_obs/GFED4.1s_Burned_Fraction.nc", \
                             "ISIMIP3a_obs/FireCCI5.1_Burned_Fraction.nc", \
                             "ISIMIP3a_obs/GFED500m_Burned_Percentage.nc"]
    filenames_observation = [dir_observation + file for file in filenames_observation]
    
    observations_names = ['GFED4.1s', 'FireCCI5.1', 'GFED500m']

    year_range = [1996, 2020]
    n_itertations = 1000
    tracesID = 'burnt_area_trace'
    mod_scale = 1.0/100.0
    obs_scale = [1.0, 1.0, 1.0/100.0]

    units = '1'
    output_file = 'outputs/trend_burnt_area_metric_results'
    output_maps = 'outputs/burnt_area/'

    region_type = 'gfed'

    log_transform = False

    run_all_eval_for_models(filenames_model, names_model, variable_model,
                            filenames_observation, observations_names,
                            year_range, 
                            n_itertations, tracesID, mod_scale,  obs_scale, units,
                            output_file, output_maps, region_type = region_type,
                            log_transform = log_transform)

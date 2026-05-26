import os
import numpy as np
import iris
import iris.coord_categorisation
import iris.quickplot as qplt
import iris.plot as iplt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pickle
import sys
sys.path.append('SoW_info/')
from state_of_wildfires_colours  import SoW_cmap
from state_of_wildfires_region_info  import get_region_info

sys.path.append('libs/')
from plot_maps import *
from  constrain_cubes_standard import *
from bilinear_interpolate_cube import *
import os.path
from pathlib import Path
from matplotlib.colors import ListedColormap, BoundaryNorm
from pdb import set_trace



def get_season_anomaly(obs_cube, year = 2024, mnths = ['06' ,'07'], diff_type = 'anomoly'):
    iris.coord_categorisation.add_month(obs_cube, 'time', name='month')
    iris.coord_categorisation.add_year(obs_cube, 'time', name='year')

    # Select months and year
    
    season = sub_year_months(obs_cube, mnths)
    season = season.aggregated_by('year', iris.analysis.SUM) 
    season_year = sub_year_range(season, [year, year])
    
    # Climatology: all years 
    clim_mean = season.collapsed('time', iris.analysis.MEAN)

    # Anomaly
    
    if diff_type == 'ratio':
        anomaly = season_year / clim_mean
        anomaly.data[clim_mean.data == 0.0] = 1.0
        anomaly.data.mask = clim_mean.data.mask
    elif diff_type == 'anomoly':
        anomaly = season_year - clim_mean
    else:
        anomaly = season_year
    return anomaly

def compute_fraction_below_obs(anomaly_cube, obs_cube):
    # Step 1: Regrid obs to match anomaly (you may want conservative or bilinear regridding)
    obs_on_model_grid = obs_cube.regrid(anomaly_cube[0], iris.analysis.Linear())

    # Step 2: Broadcast obs to shape (realization, lat, lon)
    n_realizations = anomaly_cube.shape[0]
    obs_broadcast = np.broadcast_to(obs_on_model_grid.data, anomaly_cube.shape)

    # Step 3: Compare
    comparison = anomaly_cube.data < obs_broadcast  # Boolean array

    # Step 4: Fraction across realizations
    fraction = comparison.mean(axis=0)  # shape: (lat, lon)

    # Step 5: Make a new cube
    fraction_cube = anomaly_cube[0].copy()
    fraction_cube.data = fraction
    fraction_cube.long_name = 'fraction_below_observation'
    fraction_cube.units = '1'

    return fraction_cube, obs_on_model_grid

def compute_fraction_above_threshold(anomaly_cube, maskv=0):
    """
    Returns a cube with the fraction of ensemble members where the anomaly exceeds `maskv`.
    
    Parameters:
        anomaly_cube (iris.cube.Cube): A cube with 'realization' as the first dimension.
        maskv (float): The threshold value to compare against.

    Returns:
        iris.cube.Cube: A cube with shape (lat, lon) showing the fraction above threshold.
    """
    # Step 1: Compare
    comparison = anomaly_cube.data > maskv  # Boolean array: True where anomaly > maskv

    # Step 2: Mean over realizations
    fraction = comparison.mean(axis=0)

    # Step 3: Create result cube
    fraction_cube = anomaly_cube[0].copy()
    fraction_cube.data = fraction
    fraction_cube.long_name = f'fraction_above_{maskv}'
    fraction_cube.units = '1'

    return fraction_cube

def load_ensemble_summary(paths, year  = 2024, mnths = ['06' , '07'], diff_type = 'anomoly',
                          nensemble = 0,
                          percentile=(10, 50, 90),
                          compare_vs = None,
                          return_ensemble = False,
                          obs = None):
    if not isinstance(paths, list):
        paths = [paths]

    def open_path(path):
        files = [os.path.join(path, f) for f in os.listdir(path) \
                       if f.endswith('.nc') and 'sample-pred' in f]
        if nensemble > 0:
            files = files[0:len(files):round(len(files)/nensemble)]
        
        cubes = iris.cube.CubeList([iris.load_cube(f) for f in sorted(files)])
        ensemble = cubes.merge_cube()
        
        try:
            iris.coord_categorisation.add_month(ensemble, 'time', name='month')
        except:
            pass
    
        season = sub_year_months(ensemble, mnths)
        return season
    season = [open_path(path) for path in paths]
    if len(season) == 1: 
        season = season[0]
    else:
        for sea in season[1:]: season[0].data = sea.data * season[0].data
        season = season[0]**(1/len(season))
        
    season = season.aggregated_by('year', iris.analysis.MEAN)
     
    season_year = sub_year_range(season, [year, year])
    clim_mean = season.collapsed('time', iris.analysis.MEAN)
    maskv = 0.0
    nullv = 0.0
    scale = 100.0
    compare_cube = None
    if diff_type == 'ratio':
        anomaly = season_year / clim_mean
        maskv = 1.0
        nullv = 1.0
        scale = 1.0
        
        if compare_vs is not None:
            
            anomaly = anomaly - 1.0
            scale = load_ensemble_summary(compare_vs, year, mnths, diff_type,  nensemble,                                                 compare_vs = None, return_ensemble = True) - 1.0
            mask = scale.data < 0
            scale = 1.0/scale
            scale.data[mask] = 0.0
            nullv = 0.0
    elif diff_type == 'anomoly':
        anomaly = season_year - clim_mean
    else:
        anomaly = season_year
        nullv = 0.5
    if return_ensemble: return anomaly
    pvs = compute_fraction_above_threshold(anomaly, nullv)
    anomaly *= scale
    pcs = anomaly.collapsed('realization', iris.analysis.PERCENTILE, percent=percentile)
    #pcs.data *= scale
    
    if obs is None: 
        return pcs, pvs
    else:
        obs_pos, obs_regrid = compute_fraction_below_obs(anomaly, obs)
        obs_pos.data[obs_regrid.data == maskv] = np.nan
    
        return pcs, pvs, obs_pos

def get_positive_count_layer(anom_list, threshold=0.1):
    count_cube = anom_list[0].copy()
    data = np.zeros(count_cube.shape, dtype=int)
    
    for cube in anom_list:
        data += (cube.data > threshold).astype(int)
    
    count_cube.data = np.ma.masked_array(data, mask=cube.data.mask)
    return count_cube

def open_mod_data(region_info, limitation_type = "Standard_", nensemble = 100, 
                  diff_type = 'anomoly', sow_controls = False, *args, **kw):

    rdir = region_info['dir']
    year = region_info['years'][0]
    mnths = region_info['mnths']
     
    # Load control and each perturbed scenario
    base_path = f"outputs/outputs_scratch/ConFLAME_nrt-drivers10/" + \
                rdir + "-2425/samples/_21-frac_points_0.5/baseline-"

    temp_path = "temp2/control_anom_maps7/"
    os.makedirs(temp_path, exist_ok=True)
    
    extra_path = rdir + '/' + limitation_type + '/' + str(diff_type) + '/'

    os.makedirs(temp_path + extra_path, exist_ok=True) 
    extra_path = extra_path  + "ensemble_no_" + \
                 str(nensemble) + '_year' + \
                 str(year) + '_months'.join(mnths) 
    
    temp_path = temp_path + extra_path + '.pckl'
    #set_trace()
    if os.path.isfile(temp_path):# and False:
        obs_anomaly, mod_pcs, mod_pvs, obs_pos, anom_summery, anom_summery_sow, \
            count_pos, count_neg \
            = pickle.load(open(temp_path,"rb"))
    else:
        obs = iris.load_cube("data/data/driving_data2425/" + rdir + "/burnt_area.nc")
        obs_anomaly = get_season_anomaly(obs, year, mnths, diff_type = diff_type)

        mod_pcs, mod_pvs, obs_pos = load_ensemble_summary(f"{base_path}/Evaluate", 
                                                 year, mnths, diff_type, nensemble,
                                                 obs = obs_anomaly, 
                                                 *args, **kw)

        anom_summery = [load_ensemble_summary(f"{base_path}/{limitation_type}{i}", 
                                                     year, mnths, diff_type, nensemble,
                                                     compare_vs = f"{base_path}/Evaluate",
                                                     *args, **kw) for i in range(6)]
        
        anom_summery_sow = [load_ensemble_summary([f"{base_path}/{limitation_type}{i}", 
                                                   f"{base_path}/{limitation_type}{j}"],
                                                     year, mnths, diff_type, nensemble,
                                                     compare_vs = f"{base_path}/Evaluate",
                                                     *args, **kw) \
                                                    for i,j in zip([0, 2, 4], [1, 3, 5])]
        
        pvs = [anom[1] for anom in anom_summery]
        count_pos = get_positive_count_layer(pvs, 0.1)
        count_neg = get_positive_count_layer(pvs, 0.9)
        count_neg.data = len(pvs) - count_neg.data
        
        pickle.dump([obs_anomaly, mod_pcs, mod_pvs, obs_pos, 
                     anom_summery, anom_summery_sow, count_pos, count_neg], 
                    open(temp_path, "wb"))

    if sow_controls: anom_summery = anom_summery_sow
    return obs_anomaly, mod_pcs, mod_pvs, obs_pos, \
                     anom_summery, \
                     count_pos, count_neg, extra_path, temp_path


def run_for_region(region_info, diff_type = "anomoly",
                   levels_mod = [-1, -0.1, -0.01, -0.001, 0.001, 0.01, 0.1, 1],
                   levels_controls = None, 
                   consistent = True, plot_stuff = True,
                   *args, **kw):

    obs_anomaly, mod_pcs, mod_pvs, obs_pos, anom_summery,\
            count_pos, count_neg, \
            extra_path, temp_path = open_mod_data(region_info, diff_type = diff_type,     
                                                  *args, **kw)
    
    anom_p10 = [anom[0][0] for anom in anom_summery]
    anom_p90 = [anom[0][-1] for anom in anom_summery]
    rt = None
    n_levels = 5
    force0 = True
    levels_BA_obs = None
    if diff_type == "ratio":
        rt = 1.0
        levels_BA_obs = region_info['Ratio_levels']
    if diff_type == "absolute":
        n_levels = 7
        force0 = False
    if diff_type == "anomoly":
        levels_BA_obs = region_info['Anomoly_levels']
    
    if consistent:
        levels_BA = auto_pretty_levels([obs_anomaly, mod_pcs[0], mod_pcs[-1]], 
                                       n_levels = n_levels + 3, ratio = rt) 
        levels_BA_obs = levels_BA
    else:
        levels_BA = levels_BA_obs

    
    if levels_controls is None:
        levels_controls = auto_pretty_levels(anom_p10 + anom_p90, n_levels = n_levels, 
                                             ratio = rt)
    
    # Define grid shape
    fig, axes = set_up_sow_plot_windows(5, 4, mod_pcs[0])

    smoothed_obs = smooth_cube(obs_anomaly, sigma=2)
    img = []
   
    img.append(plot_map_sow(obs_anomaly, "Observed Burned Area", 
                    cmap=SoW_cmap['diverging_TealOrange'], 
                    levels=levels_BA_obs,#region_info['Anomoly_levels'], 
                    ax=axes[0], cbar_label = "Burned Area Anomaly (%)"))
    
    img.append(plot_map_sow(mod_pcs[0], "Simulated Burned Area (10th percentile)", 
                    cmap=SoW_cmap['diverging_TealOrange'], levels=levels_BA,#levels_mod, 
                    ax=axes[4]))

    img.append(plot_map_sow(mod_pcs[-1], "Simulated Burned Area (90th percentile)",     
                    cmap=SoW_cmap['diverging_TealOrange'], levels=levels_BA,#levels_mod, 
                    ax=axes[5]))
    
    img.append(plot_map_sow(count_pos, "No. anonomlously high controls", 
                        levels = range( count_pos.data.max() + 2), 
                        cmap=SoW_cmap['gradient_hues'], extend = 'neither', ax = axes[6]))
    img.append(plot_map_sow(count_neg, "No. anonomlously low controls", 
                        levels = range( count_neg.data.max() + 2), 
                        cmap=SoW_cmap['gradient_reversed_hues'], extend = 'neither', ax = axes[7]))

    control_names = ['Fuel', 'Moisture', 'Weather', 'Wind', 'Ignitions', 'Suppression']
    cmaps = [SoW_cmap['diverging_GreenPink'].reversed(), 
            SoW_cmap['diverging_TealPurple'], 
            SoW_cmap['diverging_BlueRed'], 
            SoW_cmap['diverging_BlueRed'], 
            SoW_cmap['diverging_GreenPurple'], SoW_cmap['diverging_GreenPurple']]
    
    for i in range(len(anom_p10)):
        if not consistent: 
            levels_controls = auto_pretty_levels(anom_p10[i].data, n_levels = n_levels+1, 
                                                    ratio = rt, force0 = force0)
        img.append(plot_map_sow(anom_p10[i], control_names[i] + " (10th percentile)", 
                    cmap=cmaps[i], levels=levels_controls, 
                    ax=axes[2*i+8]))

        if not consistent: 
            levels_controls = auto_pretty_levels(anom_p90[i].data, n_levels = n_levels+1, 
                                                    ratio = rt, force0 = force0)

        img.append(plot_map_sow(anom_p90[i], control_names[i] + " (90th percentile)", 
                    cmap=cmaps[i], levels=levels_controls, 
                    ax=axes[2*i+9]))

    plt.tight_layout()
    if not consistent:
        extra_path = extra_path + 'own_levels'
    fname = "figs/control_maps_for/" + extra_path + ".png"
    path = Path(fname).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(fname, dpi=300)

    fig, axes = set_up_sow_plot_windows(1, 3, mod_pcs[0])

    img.append(plot_map_sow(obs_anomaly, "Burned Area", 
                    cmap=SoW_cmap['diverging_TealOrange'], 
                    levels=levels_BA_obs, 
                    ax=axes[0], cbar_label = "Burned Area Anomaly (%)"))


    img.append(plot_map_sow(count_pos, "Number of positive fire indicators", 
                        levels = range( count_pos.data.max() + 2), 
                        cmap=SoW_cmap['gradient_hues'], extend = 'neither', ax = axes[1]))

    img.append(plot_map_sow(count_neg, "Number of negative fire indicators", 
                        levels = range( count_neg.data.max() + 2), 
                        cmap=SoW_cmap['gradient_reversed_hues'], extend = 'neither', ax = axes[2]))

    plt.tight_layout()
    
    fname = "figs/control_maps_for/" + extra_path  + "-summer.png"
    plt.savefig(fname,  dpi=300)
    return temp_path


def show_main_control(region, control_names, cmaps, dcmaps, *args, **kw):
    region_info = get_region_info(region)[region]

    obs_anomaly, mod_pcs, mod_pvs, obs_pos, anom_summery, \
            count_pos, count_neg, \
            extra_path, temp_path = open_mod_data(region_info, 
                                                  limitation_type = "Standard_",
                                                  diff_type = "ratio", *args, **kw)
     
    fig, axes = set_up_sow_plot_windows(5, 3, mod_pcs[0], size_scale = 6)
    levels_BA_obs = region_info['Ratio_levels']

    def plot_BA(cube, label, axi, *args, **kw):
        plot_map_sow(cube, label, 
                     cmap=SoW_cmap['diverging_TealOrange'], 
                     levels=[0] + levels_BA_obs,#region_info['Anomoly_levels'], 
                     ax=axes[axi], cbar_label = "Burned Area Anomaly (ratio)",
                     extend = 'max', 
                     overlay_value = 1.0, overlay_col = "#ffffff", *args, **kw)
     
    plot_BA(obs_anomaly, "Observed Burned Area", 0)
    
    plot_BA(mod_pcs[1], "Simulated Burned Area", 1, cube_pvs = mod_pvs)

    obs_anomaly, mod_pcs, mod_pvs, obs_pos, anom_summery, \
            count_pos, count_neg, \
            extra_path, temp_path = open_mod_data(region_info, 
                                                  limitation_type = "Potential_climatology",
                                                  diff_type = "anomoly", *args, **kw)
    
    plot_map_sow(count_pos, "Number of positive fire indicators", 
                        levels = range( count_pos.data.max() + 2), 
                        cmap=SoW_cmap['gradient_hues'], extend = 'neither', ax = axes[2])

    obs_anomaly, mod_pcs, mod_pvs, obs_pos, anom_summery, \
            count_pos, count_neg, \
            extra_path, temp_path = open_mod_data(region_info, 
                                                  limitation_type = "Standard_",
                                                  diff_type = "absolute", *args, **kw)   

    def plot_control(i, scale, axis_diff, levels, extend = 'max', shift = 0.0, powr = 1.0, 
                     custom_pv = False, *args, **kw):
        cube2plot = anom_summery[i][0][1] 
        cube2plot.data *= scale
        cube2plot.data += shift
        cube2plot.data = cube2plot.data**powr
        
        if custom_pv:
            cube_pvs = (anom_summery[i][0][2]-anom_summery[i][0][0])/anom_summery[i][0][1]
            cube_pvs.data =  (100.0-cube_pvs.data)/100
            #set_trace()
        else:
            cube_pvs = anom_summery[i][1] 

        if extend == 'max':
            if np.nanmean(cube2plot.data>90) > 0.5:
                levels = 100 - np.array(levels)
                levels = np.sort(levels)
                extend = 'min'
            elif  np.nanmean(cube2plot.data>50) > 0.5:
                levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                extend = 'neither'
        levels = None
        #set_trace()
        plot_map_sow(cube2plot, control_names[i], 
                    cmap=cmaps[i], levels=levels, cube_pvs = cube_pvs,
                    ax=axes[i + axis_diff], extend = extend, levels_greater_zero = True,
                    *args, **kw)
    for i in range(len(anom_summery)):
        plot_control(i, 1, 3,
                     #[0, 0.1, 0.2, 0.4, 0.8, 1, 2, 5, 10, 20, 50],
                     [0, 0.5, 1, 5, 10, 15, 20, 30, 40, 50],
                     custom_pv = True)

    cmaps = dcmaps

    obs_anomaly, mod_pcs, mod_pvs, obs_pos, anom_summery, \
            count_pos, count_neg, \
            extra_path, temp_path = open_mod_data(region_info, 
                                                  limitation_type = "Standard_",
                                                  diff_type = "ratio", *args, **kw)
      
    for i in range(len(anom_summery)):
        plot_control(i, 1, 3 + len(anom_summery),
                     #[-100, -60, -40, -20, -10, -5, -2, -1, 0, 2, 5, 10, 20, 40, 60, 100],
                     [0, 1/8, 1/5, 1/4, 1/2 , 1, 2, 4, 5, 8],
                     shift = 1.0,
                     powr = 6.0, custom_pv = False, 
                     overlay_value = 1.0, overlay_col = "#ffffff", extend = 'max') 

    fname = "figs/control_maps_for/" + region_info['dir'] + '/' + extra_path.split('/')[-1]  + "-contol_summery.png"
    
    plt.tight_layout()
    plt.savefig(fname,  dpi=300)

levels_controls = [[1, 2, 5, 10, 20, 50, 80, 90, 95, 98, 99], None, None]
regions = ["Amazon", "Pantanal", "LA", "Congo"]
regions_info = get_region_info(regions)


control_names = ['Fuel', 'Moisture', 'Weather', 'Wind', 'Ignitions', 'Suppression']
cmaps = [SoW_cmap['gradient_teal'], 
         SoW_cmap['gradient_teal'], 
         SoW_cmap['gradient_red'], 
         SoW_cmap['gradient_red'], 
         SoW_cmap['gradient_purple'], 
         SoW_cmap['gradient_purple']]

dcmaps = [SoW_cmap['diverging_GreenPink'].reversed(), 
          SoW_cmap['diverging_TealPurple'], 
          SoW_cmap['diverging_BlueRed'], 
          SoW_cmap['diverging_BlueRed'], 
          SoW_cmap['diverging_GreenPurple'], 
          SoW_cmap['diverging_GreenPurple']]

for region in regions:
    show_main_control(region, control_names, cmaps, dcmaps, sow_controls = False)

control_names = ['Fuel', 'Weather', 'Human & Ignitions']
cmaps = [SoW_cmap['gradient_teal'], 
         SoW_cmap['gradient_red'],  
         SoW_cmap['gradient_purple']]

dcmaps = [SoW_cmap['diverging_GreenPink'].reversed(), 
          SoW_cmap['diverging_BlueRed'], 
          SoW_cmap['diverging_GreenPurple']]

for region in regions:
    show_main_control(region, control_names, cmaps, dcmaps, sow_controls = True)

set_trace()
for consistent in [False, True]:
    for region in regions:
        for diff_type, levels_control in zip(['ratio', 'absolute', 'anomoly'], levels_controls):
            for limitation_type in ["Standard_", "Potential_climateology"]:
                tfile = run_for_region(regions_info[region], diff_type = diff_type, 
                               limitation_type = limitation_type, 
                               levels_controls = levels_control,
                               consistent = consistent, plot_stuff = False)


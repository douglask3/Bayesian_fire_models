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


def load_ensemble_summary(path, year  = 2024, mnths = ['06' , '07'], diff_type = 'anomoly',
                          nensemble = 0,
                          percentile=(10, 90)):
    files = [os.path.join(path, f) for f in os.listdir(path) \
                    if f.endswith('.nc') and 'sample-pred' in f]
    if nensemble > 0:
        files = files[0:len(files):round(len(files)/nensemble)]
    
    cubes = iris.cube.CubeList([iris.load_cube(f) for f in sorted(files)])

    # Concatenate across fake ensemble dim (we'll add one)
    #for i, cube in enumerate(cubes):
    #    cube.add_aux_coord(iris.coords.AuxCoord(i, long_name='realization'))
    ensemble = cubes.merge_cube()
    
    try:
        iris.coord_categorisation.add_month(ensemble, 'time', name='month')
    except:
        pass
    
    season = sub_year_months(ensemble, mnths)
    season = season.aggregated_by('year', iris.analysis.SUM)
     
    season_year = sub_year_range(season, [year, year])
    clim_mean = season.collapsed('time', iris.analysis.MEAN)
    
    if diff_type == 'ratio':
        anomaly = season_year / clim_mean
    elif diff_type == 'anomoly':
        anomaly = season_year - clim_mean
    else:
        anomaly = season_year
    #
    p10 = anomaly.collapsed('realization', iris.analysis.PERCENTILE, percent=percentile[0])
    p90 = anomaly.collapsed('realization', iris.analysis.PERCENTILE, percent=percentile[1])
    p10.data *= 100
    p90.data *= 100
    
    return p10, p90

def get_positive_count_layer(anom_list, threshold=0):
    count_cube = anom_list[0].copy()
    data = np.zeros(count_cube.shape, dtype=int)

    for cube in anom_list:
        data += (cube.data > threshold).astype(int)
    
    count_cube.data = np.ma.masked_array(data, mask=cube.data.mask)
    return count_cube

def open_mod_data(region_info, limitation_type = "Standard_", nensemble = 100, 
                  diff_type = 'anomoly', *args, **kw):

    rdir = region_info['dir']
    year = region_info['years'][0]
    mnths = region_info['mnths']
    
    obs = iris.load_cube("data/data/driving_data2425/" + rdir + "/burnt_area.nc")
    obs_anomaly = get_season_anomaly(obs, year, mnths, diff_type = diff_type)
    
    # Load control and each perturbed scenario
    base_path = f"outputs/outputs_scratch/ConFLAME_nrt-drivers3/" + \
                rdir + "-2425/samples/_21-frac_points_0.5/baseline-"

    temp_path = "temp2/control_anom_maps/"
    os.makedirs(temp_path, exist_ok=True)
    
    extra_path = rdir + '/' + limitation_type + '/' + str(diff_type) + '/'

    os.makedirs(temp_path + extra_path, exist_ok=True) 
    extra_path = extra_path  + "ensemble_no_" + \
                 str(nensemble) + '_year' + \
                 str(year) + '_months'.join(mnths) 
    
    temp_path = temp_path + extra_path + '.pckl'
    #set_trace()
    if os.path.isfile(temp_path):
        obs_anomaly, mod_p10, mod_p90, anom_p10, anom_p90, count_pos, count_neg \
            = pickle.load(open(temp_path,"rb"))
    else:
        mod_p10, mod_p90 = load_ensemble_summary(f"{base_path}/Evaluate", 
                                                 year, mnths, diff_type, nensemble,
                                                 *args, **kw)

        anom_p90, anom_p10 = [], []
        
        for i in range(6):
            out_p10, out_p90 = load_ensemble_summary(f"{base_path}/{limitation_type}{i}", 
                                                     year, mnths, diff_type, nensemble,
                                                     *args, **kw)
            anom_p90.append(out_p90)
            anom_p10.append(out_p10)

        if diff_type == 'ratio':
            threshold = 100.0
        elif diff_type == 'anomoly':
            threshold = 0.0
        elif diff_type == 'absolute':
            threshold = 50
        count_pos = get_positive_count_layer(anom_p10, threshold)
        count_neg = get_positive_count_layer(anom_p90, threshold)
        count_neg.data = len(anom_p90) - count_neg.data
        pickle.dump([obs_anomaly, mod_p10, mod_p90, anom_p10, anom_p90, count_pos, count_neg], open(temp_path, "wb"))
    return obs_anomaly, mod_p10, mod_p90, anom_p10, anom_p90, count_pos, count_neg, extra_path


def run_for_region(region_info, diff_type = "anomoly",
                   levels_mod = [-1, -0.1, -0.01, -0.001, 0.001, 0.01, 0.1, 1],
                   levels_controls = None, 
                   consistent = True, # [-50, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50]
                   *args, **kw):
    #set_trace()
    obs_anomaly, mod_p10, mod_p90, anom_p10, anom_p90, \
        count_pos, count_neg, extra_path = open_mod_data(region_info, diff_type = diff_type,                                                    *args, **kw)
    
    rt = None
    n_levels = 5
    force0 = True
    levels_BA_obs = None
    if diff_type == "ratio":
        rt = 100.0
        levels_BA_obs = region_info['Ratio_levels']
        mod_p10 = mod_p10/100.0
        mod_p90 = mod_p90/100
    if diff_type == "absolute":
        n_levels = 7
        force0 = False
    if diff_type == "anomoly":
        levels_BA_obs = region_info['Anomoly_levels']
    
    if consistent:
        levels_BA = auto_pretty_levels([obs_anomaly, mod_p10, mod_p90], n_levels = n_levels + 3,
                                   ratio = rt) 
        levels_BA_obs = levels_BA
    else:
        levels_BA = levels_BA_obs

    if levels_controls is None:
        levels_controls = auto_pretty_levels(anom_p10 + anom_p90, n_levels = n_levels, 
                                             ratio = rt)
    
    # Define grid shape
    fig, axes = set_up_sow_plot_windows(5, 4, mod_p10)

    smoothed_obs = smooth_cube(obs_anomaly, sigma=2)
    img = []
   
    img.append(plot_map_sow(obs_anomaly, "Observed Burned Area", 
                    cmap=SoW_cmap['diverging_TealOrange'], 
                    levels=levels_BA_obs,#region_info['Anomoly_levels'], 
                    ax=axes[0], cbar_label = "Burned Area Anomaly (%)"))
    
    img.append(plot_map_sow(mod_p10, "Simulated Burned Area (10th percentile)", 
                    cmap=SoW_cmap['diverging_TealOrange'], levels=levels_BA,#levels_mod, 
                    ax=axes[4]))

    img.append(plot_map_sow(mod_p90, "Simulated Burned Area (90th percentile)",     
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

    fig, axes = set_up_sow_plot_windows(1, 3, mod_p10)

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
    

from state_of_wildfires_region_info  import get_region_info

levels_controls = [[1, 2, 5, 10, 20, 50, 80, 90, 95, 98, 99], None, None]
regions = ["Congo", "Amazon", "Pantanal", "LA"]
regions_info = get_region_info(regions)


for region in regions:
    for consistent in [False, True]:
        for diff_type, levels_control in zip(['absolute', 'anomoly', 'ratio'], levels_controls):
            if diff_type== 'ratio':
                for type in ["Standard_", "Potential"]:
                    run_for_region(regions_info[region], diff_type = diff_type, 
                               limitation_type = type, levels_controls = levels_control,
                               consistent = consistent)

set_trace()




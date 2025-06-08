import os
import numpy as np
import iris
import iris.coord_categorisation
import iris.quickplot as qplt
import iris.plot as iplt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import pickle
import sys
sys.path.append('SoW_info/')
from state_of_wildfires_colours  import SoW_cmap
from state_of_wildfires_region_info  import get_region_info

sys.path.append('libs/')
from  constrain_cubes_standard import *
from bilinear_interpolate_cube import *
import os.path
from matplotlib.colors import ListedColormap, BoundaryNorm
from pdb import set_trace


def get_cube_extent(cube):
    lon_min = cube.coord('longitude').points.min()
    lon_max = cube.coord('longitude').points.max()
    lat_min = cube.coord('latitude').points.min()
    lat_max = cube.coord('latitude').points.max()
    return [lon_min, lon_max, lat_min, lat_max]


def get_season_anomaly(obs_cube, year = 2024, mnths = ['06' ,'07'], ratio = False):
    iris.coord_categorisation.add_month(obs_cube, 'time', name='month')
    iris.coord_categorisation.add_year(obs_cube, 'time', name='year')

    # Select months and year
    
    season = sub_year_months(obs_cube, mnths)
    season = season.aggregated_by('year', iris.analysis.SUM) 
    season_year = sub_year_range(season, [year, year])
    
    # Climatology: all years 
    clim_mean = season.collapsed('time', iris.analysis.MEAN)

    # Anomaly
    if ratio:
        anomaly = season_year / clim_mean
        anomaly.data[clim_mean.data == 0.0] = 0.0
        anomaly.data.mask = clim_mean.data.mask
    else:
        anomaly = season_year - clim_mean
    return anomaly


def load_ensemble_summary(path, year  = 2024, mnths = ['06' , '07'], ratio = False,
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
    
    if ratio:
        season_year = season_year / clim_mean
    else:
        season_year = season_year - clim_mean
    
    p10 = season_year.collapsed('realization', iris.analysis.PERCENTILE, percent=percentile[0])
    p90 = season_year.collapsed('realization', iris.analysis.PERCENTILE, percent=percentile[1])
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

def open_mod_data(region_info, limitation_type = "Standard_", nensemble = 10, ratio = False, *args, **kw):

    rdir = region_info['dir']
    year = region_info['years'][0]
    mnths = region_info['mnths']

    obs = iris.load_cube("data/data/driving_data2425/" + rdir + "/burnt_area.nc")
    obs_anomaly = get_season_anomaly(obs, year, mnths, ratio = ratio)
    
    # Load control and each perturbed scenario
    base_path = f"outputs/outputs_scratch/ConFLAME_nrt-drivers3/" + \
                rdir + "-2425/samples/_21-frac_points_0.5/baseline-"

    temp_path = "temp2/control_anom_maps/"
    os.makedirs(temp_path, exist_ok=True)
    extra_path = rdir + limitation_type + str(nensemble) + \
                 str(year) + '_'.join(mnths) + str(ratio)
    temp_path = temp_path + extra_path + '.pckl'
    #set_trace()
    if os.path.isfile(temp_path):
        obs_anomaly, mod_p10, mod_p90, anom_p10, anom_p90, count_pos, count_neg \
            = pickle.load(open(temp_path,"rb"))
    else:
        mod_p10, mod_p90 = load_ensemble_summary(f"{base_path}/Evaluate", 
                                                 year, mnths, ratio, nensemble,
                                                 *args, **kw)

        anom_p90, anom_p10 = [], []
        
        for i in range(6):
            out_p10, out_p90 = load_ensemble_summary(f"{base_path}/{limitation_type}{i}", 
                                                     year, mnths, ratio, nensemble,
                                                     *args, **kw)
        
            anom_p90.append(out_p90)
            anom_p10.append(out_p10)
        if ratio:
            threshold = 1.0
        else:
            threshold = 0.0
        count_pos = get_positive_count_layer(anom_p10, threshold)
        count_neg = get_positive_count_layer(anom_p90, threshold)
        count_neg.data = len(anom_p90) - count_neg.data
        pickle.dump([obs_anomaly, mod_p10, mod_p90, anom_p10, anom_p90, count_pos, count_neg], open(temp_path, "wb"))
    return obs_anomaly, mod_p10, mod_p90, anom_p10, anom_p90, count_pos, count_neg, extra_path

import numpy as np

import numpy as np

def concat_cube_data(cubes):
    """
    Concatenate data from a list of Iris cubes into one flat NumPy array,
    skipping any masked/invalid data.
    """
    data_list = []

    for cube in cubes:
        data = np.ma.masked_invalid(cube.data)
        if np.ma.is_masked(data):
            valid_data = data.compressed()  # only unmasked values
        else:
            valid_data = data.ravel()
        data_list.append(valid_data)

    return np.concatenate(data_list)


def auto_pretty_levels(data, n_levels=7, log_ok=True):
    """
    Generate 'pretty' contour levels that break the data into roughly equal-sized areas.

    Parameters:
    - data: array-like or Iris cube (flattened automatically)
    - n_levels: desired number of contour bins (not including endpoints)
    - log_ok: allow log scaling if data is skewed and positive

    Returns:
    - levels: list of nicely rounded level values
    """
    try:
        data = concat_cube_data(data)
    except:
        pass
    # Flatten data and mask NaNs
    data = np.ma.masked_invalid(np.ravel(data))
    if data.mask.all():
        raise ValueError("No valid data found to calculate levels.")

    # Try log-scale if distribution is strongly right-skewed and positive
    if log_ok and np.all(data > 0) and (np.percentile(data, 90) / np.percentile(data, 10)) > 50:
        log_data = np.log10(data)
        percentiles = np.linspace(0, 100, n_levels + 1)
        levels_raw = np.power(10, np.percentile(log_data, percentiles))
    else:
        percentiles = np.linspace(0, 100, n_levels + 1)
        levels_raw = np.percentile(data, percentiles)

    # Round levels to 'nice' numbers
    def nice_round(x):
        if x == 0:
            return 0
        magnitude = 10 ** np.floor(np.log10(abs(x)))
        mantissa = round(x / magnitude, 1)
        return mantissa * magnitude

    levels_rounded = sorted(set([nice_round(lv) for lv in levels_raw]))

    # Ensure levels are strictly increasing and unique
    while len(levels_rounded) <= 2 and n_levels < 20:
        n_levels += 2  # try with more bins if too few unique rounded levels
        return auto_pretty_levels(data, n_levels=n_levels, log_ok=log_ok)

    levels_rounded = np.array(levels_rounded)
    if (any(levels_rounded) < 0 and any(data) > 0) or \
            (any(levels_rounded) > 0 and any(data) < 0):
        levels_rounded = np.sort(np.unique(np.append(levels_rounded, - levels_rounded)))
    
    return levels_rounded


def plot_map(cube, title='', contour_obs=None, cmap='RdBu_r', 
             levels = None, extend = 'both', ax=None,
             cbar_label = ''):
    
    cube.long_name = title
    cube.rename(title)
    is_catigorical =  np.issubdtype(cube.core_data().dtype, np.integer)
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))
    # Main filled contour
    if levels is  None:
        levels = auto_pretty_levels(cube.data)
    if is_catigorical:
        norm = BoundaryNorm(boundaries=np.array(levels) + 0.5, ncolors=cmap.N)
    else:   
        norm = BoundaryNorm(boundaries=levels,  ncolors=cmap.N, extend = extend)
        
    img = iplt.contourf(cube, levels=levels, cmap=cmap, axes=ax, extend = extend, 
                        norm = norm)
    #else:
    #    img = iplt.pcolormesh(cube, cmap=cmap, axes=ax)
    if is_catigorical:
        tick_positions = np.array(levels) + 0.5
        tick_labels = [str(level) for level in levels]
        cbar = plt.colorbar(img, ax=ax, orientation='horizontal',
                            ticks=tick_positions)
        cbar.ax.set_xticklabels(tick_labels) 
    else:
        cbar = plt.colorbar(img, ax=ax, ticks=levels, orientation='horizontal')
    cbar.set_label(cbar_label, labelpad=10, loc='center')
    cbar.ax.xaxis.set_label_position('top')
     
    # Add boundaries
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    # Optional observed burned area anomaly contour
    if contour_obs is not None:
        qplt.contour(contour_obs, levels=[0], colors='#8a3b00', linewidths=1, axes=ax)

    print(title)
    ax.set_title(title)
    return img

def run_for_region(region_info, 
                   levels_mod = [-1, -0.1, -0.01, -0.001, 0.001, 0.01, 0.1, 1],
                   levels_controls = [-50, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50],
                   *args, **kw):
    
    obs_anomaly, mod_p10, mod_p90, anom_p10, anom_p90, \
        count_pos, count_neg, extra_path= open_mod_data(region_info, *args, **kw)
    
    # Define grid shape
    n_rows, n_cols = 5, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16), 
                             subplot_kw={'projection': ccrs.PlateCarree()})

    extent = get_cube_extent(mod_p10)
    extent[0] -= (extent[1] - extent[0])*0.1
    extent[1] += (extent[1] - extent[0])*0.1
    extent[2] -= (extent[3] - extent[2])*0.1
    extent[3] += (extent[3] - extent[2])*0.1
    
    for ax in axes.flat:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Flatten axes for easy indexing
    axes = axes.flatten()

    smoothed_obs = smooth_cube(obs_anomaly, sigma=2)
    img = []
   
    img.append(plot_map(obs_anomaly, "Observed Burned Area", 
                    cmap=SoW_cmap['diverging_TealOrange'], 
                    levels=region_info['Anomoly_levels'], 
                    ax=axes[0], cbar_label = "Burned Area Anomaly (%)"))

    img.append(plot_map(mod_p10, "Simulated Burned Area (10th percentile)", 
                    cmap=SoW_cmap['diverging_TealOrange'], levels=levels_mod, 
                    ax=axes[4]))

    img.append(plot_map(mod_p90, "Simulated Burned Area (90th percentile)", 
                    cmap=SoW_cmap['diverging_TealOrange'], levels=levels_mod, 
                    ax=axes[5]))
    
    #fig.colorbar(
    #    img[2],# Use last image from that row — assumes all share cmap/norm
     #   ax=axes[1:3],    # Both axes in the row
    #    orientation='horizontal',
    #    fraction=1.0, pad=-2  # Adjust spacing as needed
    #)
    img.append(plot_map(count_pos, "No. anonomlously high controls", 
                        levels = range( count_pos.data.max() + 2), 
                        cmap=SoW_cmap['gradient_hues'], extend = 'neither', ax = axes[6]))
    img.append(plot_map(count_neg, "No. anonomlously low controls", 
                        levels = range( count_neg.data.max() + 2), 
                        cmap=SoW_cmap['gradient_reversed_hues'], extend = 'neither', ax = axes[7]))

    control_names = ['Fuel', 'Moisture', 'Weather', 'Wind', 'Ignitions', 'Suppression']
    cmaps = [SoW_cmap['diverging_GreenPink'].reversed(), 
            SoW_cmap['diverging_TealPurple'], 
            SoW_cmap['diverging_BlueRed'], 
            SoW_cmap['diverging_BlueRed'], 
            SoW_cmap['diverging_GreenPurple'], SoW_cmap['diverging_GreenPurple']]
    #levels_controls = levels_mod#[] 
    
    levels_control = auto_pretty_levels(anom_p10 + anom_p90)
         
    
    for i in range(len(anom_p10)):
        img.append(plot_map(anom_p10[i], control_names[i] + " (10th percentile)", 
                    cmap=cmaps[i], levels=levels_controls, 
                    ax=axes[2*i+8]))

        img.append(plot_map(anom_p90[i], control_names[i] + " (90th percentile)", 
                    cmap=cmaps[i], levels=levels_controls, 
                    ax=axes[2*i+9]))

    plt.tight_layout()
    plt.savefig("figs/control_maps_for" + extra_path + ".png")

    fig, axes = plt.subplots(1, 3, figsize=(9, 4), 
                             subplot_kw={'projection': ccrs.PlateCarree()})

    for ax in axes.flat:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    img.append(plot_map(obs_anomaly, "Burned Area", 
                    cmap=SoW_cmap['diverging_TealOrange'], 
                    levels=region_info['Anomoly_levels'], 
                    ax=axes[0], cbar_label = "Burned Area Anomaly (%)"))


    img.append(plot_map(count_pos, "Number of positive fire indicators", 
                        #contour_obs = smoothed_obs,
                        levels = range( count_pos.data.max() + 2), 
                        cmap=SoW_cmap['gradient_hues'], extend = 'neither', ax = axes[1]))

    img.append(plot_map(count_neg, "Number of negative fire indicators", 
                       # contour_obs = smoothed_obs,
                        levels = range( count_neg.data.max() + 2), 
                        cmap=SoW_cmap['gradient_reversed_hues'], extend = 'neither', ax = axes[2]))

    plt.tight_layout()
    plt.savefig("figs/burning_indicators_for" + extra_path + "-2.png")
    

from state_of_wildfires_region_info  import get_region_info

regions = ["Congo", "Amazon", "Pantanal", "LA"]
regions_info = get_region_info(regions)

for ratio in [True, False]:
    for type in ["Standard_", "Potential"]:
        for region in regions:
            run_for_region(regions_info[region], limitation_type = type, ratio = ratio)

set_trace()
#fig.colorbar(img3, ax=axes[2:3], orientation='horizontal', fraction=0.05, pad=0.05)
#run_for_region()


#img4 = plot_map(mod_p10[0], "Fuel Control Anomaly (90th percentile)", contour_obs=smoothed_obs, ax=axes[4])




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


def get_jun_jul_anomaly(obs_cube, year = 2024):
    iris.coord_categorisation.add_month(obs_cube, 'time', name='month')
    iris.coord_categorisation.add_year(obs_cube, 'time', name='year')

    # Select June and July and year
    summer = sub_year_months(obs_cube, [5,6])
    summer = summer.aggregated_by('year', iris.analysis.SUM) 
    summer_last = sub_year_range(summer, [year, year+1])

    # Climatology: all years 
    clim_mean = summer.collapsed('time', iris.analysis.MEAN)

    # Anomaly
    anomaly = summer_last - clim_mean
    return anomaly


def load_ensemble_summary(path, percentile=(5, 95), year  = 2024):
    files = [os.path.join(path, f) for f in os.listdir(path) \
                    if f.endswith('.nc') and 'sample-pred' in f]
    #files = files[0:len(files):round(len(files)/100)]
    
    cubes = iris.cube.CubeList([iris.load_cube(f) for f in sorted(files)])

    # Concatenate across fake ensemble dim (we'll add one)
    #for i, cube in enumerate(cubes):
    #    cube.add_aux_coord(iris.coords.AuxCoord(i, long_name='realization'))
    ensemble = cubes.merge_cube()
    
    try:
        iris.coord_categorisation.add_month(ensemble, 'time', name='month')
    except:
        pass
    
    summer = sub_year_months(ensemble, [5,6])
    summer = summer.aggregated_by('year', iris.analysis.SUM)
    summer_last = sub_year_range(summer, [year, year+1])
    clim_mean = summer.collapsed('time', iris.analysis.MEAN)
    try:
        summer_last = summer_last - clim_mean
    except:
        summer_last.data = summer_last.data - clim_mean.data
    
    p10 = summer_last.collapsed('realization', iris.analysis.PERCENTILE, percent=percentile[0])
    p90 = summer_last.collapsed('realization', iris.analysis.PERCENTILE, percent=percentile[1])
    p10.data *= 100
    p90.data *= 100
    return p10, p90

def plot_map(cube, title='', contour_obs=None, cmap='RdBu_r', 
             levels = [-2, -1, -0.5, 0, 0.5, 1, 2], extend = 'both', ax=None,
             cbar_label = ''):

    cube.long_name   = title
    cube.rename(title)
    is_catigorical =  np.issubdtype(cube.core_data().dtype, np.integer)
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))
    # Main filled contour
    if levels is not None:
        if is_catigorical:
            norm = BoundaryNorm(boundaries=np.array(levels) + 0.5, ncolors=cmap.N)
        else:   
            norm = BoundaryNorm(boundaries=levels,  ncolors=cmap.N, extend = extend)
        
        img = iplt.contourf(cube, levels=levels, cmap=cmap, axes=ax, extend = extend, 
                            norm = norm)
    else:
        img = iplt.pcolormesh(cube, cmap=cmap, axes=ax)
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
    #    set_trace()
    #else:
    #    plt.colorbar(img, orientation='horizontal')
     
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

def get_positive_count_layer(anom_list, threshold=0):
    count_cube = anom_list[0].copy()
    data = np.zeros(count_cube.shape, dtype=int)

    for cube in anom_list:
        data += (cube.data > threshold).astype(int)
    
    count_cube.data = np.ma.masked_array(data, mask=cube.data.mask)
    return count_cube

def open_mod_data(base_path, temp_path):
    #set_trace()
    if os.path.isfile(temp_path) and False:
        mod_p10, mod_p90, anom_p10, anom_p90, count = pickle.load(open(temp_path,"rb"))
    else:
        mod_p10, mod_p90 = load_ensemble_summary(f"{base_path}/Evaluate")

        anom_p90, anom_p10 = [], []
        
        for i in range(6):
            out_p10, out_p90 = load_ensemble_summary(f"{base_path}/Standard_{i}")
        
            anom_p90.append(out_p90)
            anom_p10.append(out_p10)
    
        count = get_positive_count_layer(anom_p10)
        pickle.dump([mod_p10, mod_p90, anom_p10, anom_p90, count], open(temp_path, "wb"))
    return mod_p10, mod_p90, anom_p10, anom_p90, count



def run_for_region(region = "Congo", 
                   levels_obs = [-10, -5, -2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2, 5, 10],
                   levels_mod = [-1, -0.1, -0.01, -0.001, 0.001, 0.01, 0.1, 1]):
    # Load observed anomaly
    obs = iris.load_cube("data/data/driving_data2425/" + region + "/burnt_area.nc")
    obs_anomaly = get_jun_jul_anomaly(obs)
    
    # Load control and each perturbed scenario
    base_path = "outputs/outputs_scratch/ConFLAME_nrt-drivers3/" + \
                region + "-2425/samples/_21-frac_points_0.5/baseline-"

    temp_path = "temp2/control_anom_maps/"
    os.makedirs(temp_path, exist_ok=True)
    temp_path = temp_path + region + '.pckl'
    mod_p10, mod_p90, anom_p10, anom_p90, count = open_mod_data(base_path,  temp_path)
    
    # Define grid shape
    n_rows, n_cols = 4, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12), 
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
                    cmap=SoW_cmap['diverging_TealOrange'], levels=levels_obs, 
                    ax=axes[0], cbar_label = "Burned Area Anomaly (%)"))

    img.append(plot_map(mod_p10, "Simulated Burned Area (10th percentile)", 
                    cmap=SoW_cmap['diverging_TealOrange'], levels=levels_mod, 
                    ax=axes[1]))

    img.append(plot_map(mod_p90, "Simulated Burned Area (90th percentile)", 
                    cmap=SoW_cmap['diverging_TealOrange'], levels=levels_mod, 
                    ax=axes[2]))
    
    #fig.colorbar(
    #    img[2],# Use last image from that row — assumes all share cmap/norm
     #   ax=axes[1:3],    # Both axes in the row
    #    orientation='horizontal',
    #    fraction=1.0, pad=-2  # Adjust spacing as needed
    #)
    img.append(plot_map(count, "No. anonomlous controls", 
                        levels = range( count.data.max() + 2), 
                        cmap=SoW_cmap['gradient_hues'], extend = 'neither', ax = axes[3]))

    control_names = ['Fuel', 'Moisture', 'Weather', 'Wind', 'Ignitions', 'Suppression']
    cmaps = [SoW_cmap['diverging_GreenPink'].reversed(), 
            SoW_cmap['diverging_TealPurple'], 
            SoW_cmap['diverging_BlueRed'], 
            SoW_cmap['diverging_BlueRed'], 
            SoW_cmap['diverging_GreenPurple'], SoW_cmap['diverging_GreenPurple']]
    levels_controls = levels_mod#[] 
    for i in range(len(anom_p10)):
        img.append(plot_map(anom_p10[i], control_names[i] + " (10th percentile)", 
                    cmap=cmaps[i], levels=levels_controls, 
                    ax=axes[2*i+4]))

        img.append(plot_map(anom_p90[i], control_names[i] + " (90th percentile)", 
                    cmap=cmaps[i], levels=levels_controls, 
                    ax=axes[2*i+5]))

    plt.tight_layout()
    plt.savefig("figs/control_maps_for" + region + ".png")

    fig, axes = plt.subplots(1, 2, figsize=(6, 4), 
                             subplot_kw={'projection': ccrs.PlateCarree()})

    for ax in axes.flat:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    img.append(plot_map(obs_anomaly, "Burned Area", 
                    cmap=SoW_cmap['diverging_TealOrange'], 
                    levels=[-10, -5, -2, -1, 1, 2, 5, 10], 
                    ax=axes[0], cbar_label = "Burned Area Anomaly (%)"))

    img.append(plot_map(count, "Number of fire indicators", 
                        contour_obs = smoothed_obs,
                        levels = range( count.data.max() + 2), 
                        cmap=SoW_cmap['gradient_hues'], extend = 'neither', ax = axes[1]))

    plt.tight_layout()
    plt.savefig("figs/burning_indicators_for" + region + "-2.png")
    set_trace()
#fig.colorbar(img3, ax=axes[2:3], orientation='horizontal', fraction=0.05, pad=0.05)
run_for_region()


#img4 = plot_map(mod_p10[0], "Fuel Control Anomaly (90th percentile)", contour_obs=smoothed_obs, ax=axes[4])




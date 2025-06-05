import os
import numpy as np
import iris
import iris.coord_categorisation
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import sys
sys.path.append('SoW_info/')
from state_of_wildfires_colours  import SoW_cmap
from state_of_wildfires_region_info  import get_region_info

sys.path.append('libs/')
from  constrain_cubes_standard import *
from bilinear_interpolate_cube import *

from matplotlib.colors import ListedColormap, BoundaryNorm
from pdb import set_trace

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
    files = files[0:len(files):round(len(files)/100)]
    
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
             levels = [-2, -1, -0.5, 0, 0.5, 1, 2], extend = 'both', ax=None):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))
    # Main filled contour
    if levels is not None:
        img = qplt.contourf(cube, levels=levels, cmap=cmap, axes=ax, extend = extend, 
                            norm = BoundaryNorm(boundaries=levels, 
                                                ncolors=cmap.N, extend = extend))
    else:
        img = qplt.pcolormesh(cube, cmap=cmap, axes=ax)
    
    plt.title(title)
    #plt.colorbar(img, orientation='horizontal')

    # Add boundaries
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    # Optional observed burned area anomaly contour
    if contour_obs is not None:
        qplt.contour(contour_obs, levels=[0], colors='k', linewidths=1, axes=ax)


levels = [-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10]  # example anomaly levels

# Load observed anomaly
obs = iris.load_cube("data/data/driving_data2425/Congo/burnt_area.nc")
obs_anomaly = get_jun_jul_anomaly(obs)

# Load control and each perturbed scenario
base_path = "outputs/outputs_scratch/ConFLAME_nrt-drivers2/Congo-2425/samples/_21-frac_points_0.5/baseline-"
mod_p10, mod_p90 = load_ensemble_summary(f"{base_path}/Evaluate")

anom_p90, anom_p10 = [], []

for i in range(6):
    out_p10, out_p90 = load_ensemble_summary(f"{base_path}/Standard_{i}")
    
    anom_p90.append(out_p90)
    anom_p10.append(out_p10)

def get_positive_count_layer(anom_list, threshold=0):
    count_cube = anom_list[0].copy()
    data = np.zeros(count_cube.shape, dtype=int)

    for cube in anom_list:
        data += (cube.data > threshold).astype(int)

    count_cube.data = data
    return count_cube

count = get_positive_count_layer(anom_p10)


# Define grid shape
n_rows, n_cols = 4, 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8),
                        subplot_kw={'projection': ccrs.PlateCarree()})

# Flatten axes for easy indexing
axes = axes.flatten()

from scipy.ndimage import gaussian_filter

def smooth_cube(cube, sigma=1):
    smoothed = cube.copy()
    smoothed.data = gaussian_filter(smoothed.data, sigma=sigma)
    return smoothed

smoothed_obs = smooth_cube(obs_anomaly, sigma=2)



img1 = plot_map(count, "Count", contour_obs=smoothed_obs, levels = range(7), 
                cmap=SoW_cmap['gradient_hues'], extend = 'max', ax = axes[0])

levels = [-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10]
img0 = plot_map(obs_anomaly, "Observed Burned Area Anomaly (Jun-Jul)", 
                cmap=SoW_cmap['diverging_TealOrange'], levels=levels, 
                ax=axes[1])
levels = [-1, -0.3, -0.1, -0.01, 0.01, 0.1, 0.3, 1]
img2 = plot_map(mod_p10, "Simulated (10th percentile)", 
                cmap=SoW_cmap['diverging_TealOrange'], levels=levels, 
                ax=axes[2])

img3 = plot_map(mod_p90, "Simulated (90th percentile)", 
                cmap=SoW_cmap['diverging_TealOrange'], levels=levels, 
                ax=axes[3])
#fig.colorbar(img3, ax=axes[2:3], orientation='horizontal', fraction=0.05, pad=0.05)



#img4 = plot_map(mod_p10[0], "Fuel Control Anomaly (90th percentile)", contour_obs=smoothed_obs, ax=axes[4])

set_trace()


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


def load_ensemble_summary(path, percentile=(10, 90), year  = 2024):
    files = [os.path.join(path, f) for f in os.listdir(path) \
                    if f.endswith('.nc') and 'sample-pred' in f]
    files = files[0:len(files):round(len(files)/10)]
    
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
    
    return p10, p90

#def get_anomaly(control_cube, perturbed_cube):
#    out = perturbed_cube - control_cube
#    set_trace()
#    return out


def plot_map(cube, title='', contour_obs=None, cmap='RdBu_r', levels = [-2, -1, -0.5, 0, 0.5, 1, 2], ax=None):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))
    img = qplt.contourf(cube, levels=levels, cmap=cmap, extend='both')
    
    plt.title(title)
    plt.colorbar(img, orientation='horizontal')

    # Add boundaries
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    # Optional contour of observed anomaly
    if contour_obs is not None:
        qplt.contour(contour_obs, levels=[0], colors='k', linewidths=1.2, linestyles='--')

    plt.tight_layout()
    

def summarise_anomalies(pos_anoms, neg_anoms, threshold=0):
    # Assuming all cubes are aligned
    
    summary = pos_anoms[0].copy()
    data = np.zeros(summary.shape)

    for pos, neg in zip(pos_anoms, neg_anoms):
        data += (pos.data > threshold).astype(int)
        data -= (neg.data < -threshold).astype(int)

    # Assign: 1 = all pos, -1 = all neg, 0 = mixed or neutral
    summary.data = np.where(data == len(pos_anoms), 1, 
                   np.where(data == -len(pos_anoms), -1, 0))
    return summary

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

# Summary maps
pos_summary = summarise_anomalies(anom_p90, anom_p10)
neg_summary = summarise_anomalies(anom_p90, anom_p10)


def get_positive_count_layer(anom_list, threshold=0):
    count_cube = anom_list[0].copy()
    data = np.zeros(count_cube.shape, dtype=int)

    for cube in anom_list:
        data += (cube.data > threshold).astype(int)

    count_cube.data = data
    return count_cube

def get_bitmask_layer(anom_list, threshold=0):
    bitmask_cube = anom_list[0].copy()
    bitmask_data = np.zeros(bitmask_cube.shape, dtype=int)

    for i, cube in enumerate(anom_list):
        bit = 1 << i  # e.g., 1, 2, 4, 8, ...
        bitmask_data += (cube.data > threshold).astype(int) * bit

    bitmask_cube.data = bitmask_data
    return bitmask_cube

from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_summary_layers(count_cube, bitmask_cube, title=''):
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Grayscale shading for count of +ive controls
    img1 = qplt.pcolormesh(count_cube, cmap='Greys', vmin=0, vmax=6, axes=ax)

    # Categorical colormap: pick distinct colors for major patterns
    bit_colors = ['none', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    cmap = ListedColormap(bit_colors + ['#999999'])  # up to 8 patterns, extend as needed
    norm = BoundaryNorm(boundaries=np.arange(0, len(bit_colors)+2)-0.5, ncolors=len(bit_colors)+1)

    img2 = qplt.pcolormesh(bitmask_cube, cmap=cmap, norm=norm, axes=ax)

    plt.title(title)
    plt.colorbar(img1, ax=ax, orientation='horizontal', label='Number of +ive anomalies')
    plt.colorbar(img2, ax=ax, orientation='vertical', label='Which controls')

    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    plt.tight_layout()
    plt.show()

count = get_positive_count_layer(anom_p10)

bitmask = get_bitmask_layer(anom_p10)
#plot_summary_layers(count, bitmask, title="Summary Map: Positive Control Anomalies")

# Define grid shape
n_rows, n_cols = 3, 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12    , 8)),
                        # subplot_kw={'projection': ccrs.PlateCarree()})

# Flatten axes for easy indexing
axes = axes.flatten()

img0 = plot_map(obs_anomaly, "Observed Burned Area Anomaly (Jun-Jul)", cmap=SoW_cmap['diverging_TealOrange'], levels=levels, ax=axes[0])
img1 = plot_map(count, "Counr", contour_obs=obs_anomaly, levels = range(7), ax = axes[1])
img2 = plot_map(anom_p90[0], "Fuel Control Anomaly (90th percentile)", contour_obs=obs_anomaly, ax=axes[2])

set_trace()

plot_map(obs_anomaly, "Observed Burned Area Anomaly (Jun-Jul)", cmap=SoW_cmap['diverging_TealOrange'], levels = levels)


# Plot examples

plot_map(anom_p90[0], "Fuel Control Anomaly (90th percentile)", contour_obs=obs_anomaly)
plot_map(pos_summary, "Regions with All Controls Showing Positive Anomaly", contour_obs=obs_anomaly, cmap='PiYG')
plot_map(neg_summary, "Regions with All Controls Showing Negative Anomaly", contour_obs=obs_anomaly, cmap='PiYG')




import iris
import numpy as np
import os
from iris.coords import DimCoord
import glob

import sys
sys.path.append('SoW_info/')
from state_of_wildfires_colours  import SoW_cmap
from state_of_wildfires_region_info  import get_region_info


sys.path.append('libs/')
from plot_maps import *
from  constrain_cubes_standard import *

def load_and_average_months(file, months, years):
    """Extract and average over selected months."""
    cube = iris.load_cube(file)
    season = sub_year_months(cube, months)
    season_year = sub_year_range(season, years)
    return season_year.collapsed('time', iris.analysis.MEAN)

def plot_factual_and_cf(variable, factual_path, cf_dir, months, years, label=None, shift = 0.0):
    # Load factual
    factual_avg = load_and_average_months(factual_path, months, years) + shift
    
    # Load counterfactual ensemble
    cf_cubes = []
    files  = glob.glob(cf_dir + '**', recursive = True)
    for file in files:
        if file.endswith(".nc") and variable in file:
            cube_avg = load_and_average_months(file, months, years) + shift
            cf_cubes.append(cube_avg)

    # Concatenate across a new dimension
    if not cf_cubes:
        raise ValueError("No counterfactual files found for variable:", variable)
        
    ref_cube = factual_avg.copy()
    cf_data = np.stack([cube.data for cube in cf_cubes], axis=0)
    
    # Compute 10th, 50th, 90th percentiles across realisations
    cf_10 = ref_cube.copy(data=np.percentile(cf_data, 10, axis=0))
    cf_10.data[factual_avg.data.mask] = np.nan
    cf_90 = ref_cube.copy(data=np.percentile(cf_data, 90, axis=0))
    cf_90.data[factual_avg.data.mask] = np.nan
    diff_10 = ref_cube.copy(data=factual_avg.data - cf_10.data)
    diff_90 = ref_cube.copy(data=factual_avg.data - cf_90.data)
     
    # Plot
    title_label = label or variable
    levels = auto_pretty_levels(np.append(np.append(cf_10.data.flatten(), cf_90.data.flatten()), factual_avg.data.flatten()), ignore_v = shift)
    
    plot_map_sow(factual_avg, title=f"Factual {title_label}", 
                 cbar_label=variable, levels = levels)
    plot_map_sow(cf_10, title=f"CF 10th percentile {title_label}", 
                 cbar_label=variable, levels = levels)
    plot_map_sow(cf_90, title=f"CF 90th percentile {title_label}", 
                 cbar_label=variable, levels = levels)
    levels = auto_pretty_levels([diff_10, diff_90])
    plot_map_sow(diff_10, title=f"Difference (Factual - CF 10th percentile) {title_label}", 
                 cbar_label=f"Δ {variable}", levels = levels)
    plot_map_sow(diff_90, title=f"Difference (Factual - CF 90th percentile) {title_label}", 
                 cbar_label=f"Δ {variable}", levels = levels)


region = 'Amazon'
region_info = get_region_info(region)[region]

variable = 'tas_max'
# Variable: tas_max
plot_factual_and_cf(
    variable = variable,
    factual_path = 'data/data/driving_data2425/' + region_info['dir'] + \
                   '/nrt/era5_monthly/' + variable + '.nc',
    cf_dir = 'data/data/driving_data2425/' + region_info['dir'] + \
             '/nrt/era5_monthly/CF/',
    months = region_info['mnths'],
    years = region_info['years'],
    label='Max Temp',
    shift = -273.15
)

plt.show()
set_trace()

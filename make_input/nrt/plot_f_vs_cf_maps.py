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
    if len(months) > 1:
        season_year = season_year.collapsed('time', iris.analysis.MEAN)
    return season_year    

def plot_factual_and_cf(variable, factual_path, cf_dir, months, years, label=None, units = '',
                        shift = 0.0, scale = 1.0, 
                        cmap = 'gradient_hues', dcmap = 'diverging_TealOrange',
                        axes = None, ax0 = None):
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
    

    if axes is None:
        fig, axes = set_up_sow_plot_windows(5, 1, factual_avg)
        ax0 = 0
    cbar_label = variable + ' (' + units + ')'
    plot_map_sow(factual_avg, title=f"Factual {title_label}", cmap = cmap,
                 cbar_label=variable, levels = levels, ax = axes[ax0])
    plot_map_sow(cf_10, title=f"CF 10th percentile {title_label}", cmap = cmap,
                 cbar_label=cbar_label, levels = levels, ax = axes[ax0+1])
    plot_map_sow(cf_90, title=f"CF 90th percentile {title_label}", cmap = cmap,
                 cbar_label=cbar_label, levels = levels, ax = axes[ax0 + 2])
    levels = auto_pretty_levels([diff_10, diff_90])
    plot_map_sow(diff_10, title=f"Difference (Factual - CF 10th percentile) {title_label}", 
                 cmap = dcmap,
                 cbar_label=f"Δ {cbar_label}", levels = levels, ax = axes[ax0 + 3])
    plot_map_sow(diff_90, title=f"Difference (Factual - CF 90th percentile) {title_label}", 
                 cmap = dcmap,
                 cbar_label=f"Δ {cbar_label}", levels = levels, ax = axes[ax0 + 4])



variable_info = {'tas_mean':{"file": 'tas_mean', 'label': 'Mean Monthly Temp', 'Units': "°C", 
                            'shift': -273.15, 'scale': 1.0, 
                            'cmap': SoW_cmap['gradient_red'], 
                            'dcmap': SoW_cmap['diverging_BlueRed']},
                 'tas_max':{"file": 'tas_max', 'label': 'Max Monthly Temp', 'Units': "°C", 
                            'shift': -273.15, 'scale': 1.0, 
                            'cmap': SoW_cmap['gradient_red'], 
                            'dcmap': SoW_cmap['diverging_BlueRed']},
                 'precip': {"file": 'precip', 'label': 'Precipitation', 'Units': "mm/day",
                            "shift": 0.0, 'scale': 1.0, 
                            'cmap': SoW_cmap['gradient_teal'], 
                            'dcmap': SoW_cmap['diverging_TealOrange'].reversed()},
                 'dry_days': {"file": 'dry_days', 'label': 'Mean. no dry days', 
                              'Units': "fraction",
                              "shift": 0.0, 'scale': 1.0, 
                              'cmap': SoW_cmap['gradient_teal'].reversed(), 
                              'dcmap': SoW_cmap['diverging_TealOrange']},
                 'max_consec_dry': {"file": 'dry_days', 'label': 'Max. no consecutive dry days',
                              'Units': "no. days",
                              "shift": 0.0, 'scale': 1.0, 
                              'cmap': SoW_cmap['gradient_teal'].reversed(), 
                              'dcmap': SoW_cmap['diverging_TealOrange']},
                 'hurs_mean': {"file": 'hurs_mean', 'label': 'Humidity', 
                               'Units': "%",
                               "shift": 0.0, 'scale': 1.0, 
                               'cmap': SoW_cmap['gradient_hotpink'], 
                               'dcmap': SoW_cmap['diverging_TealPurple'].reversed()},
                 'hurs_min': {"file": 'hurs_min', 'label': 'Min. Humidity', 
                               'Units': "%",
                               "shift": 0.0, 'scale': 1.0, 
                               'cmap': SoW_cmap['gradient_hotpink'], 
                               'dcmap': SoW_cmap['diverging_TealPurple'].reversed()},
                 'wind_max': {"file": 'wind_max', 'label': 'Max. Wind', 
                               'Units': "m/s",
                               "shift": 0.0, 'scale': 1.0, 
                               'cmap': SoW_cmap['gradient_purple'], 
                               'dcmap': SoW_cmap['diverging_GreenPink'].reversed()}}

variables = ['tas_max', 'precip', 'dry_days', 'max_consec_dry', 'hurs_mean', 'hurs_min',
             'wind_max']

    
regions = ['Amazon', 'Congo', 'Pantanal']

for region in regions:
    region_info = get_region_info(region)[region]
    eg_cube = iris.load_cube('data/data/driving_data2425/' + region_info['dir'] + \
                           '/nrt/era5_monthly/' + variable_info[variables[0]]['file'] + '.nc')
    fig, axes = set_up_sow_plot_windows(len(variables), 5, 
                                        eg_cube = eg_cube, figsize = (30, 30))

    for i, variable in enumerate(variables):
        info = variable_info[variable]
        plot_factual_and_cf(
            variable = info['file'],
            factual_path = 'data/data/driving_data2425/' + region_info['dir'] + \
                           '/nrt/era5_monthly/' + variable + '.nc',
            cf_dir = 'data/data/driving_data2425/' + region_info['dir'] + \
                     '/nrt/era5_monthly/CF/',
            months = region_info['mnths'],
            years = region_info['years'],
            label = info['label'],
            units = info['Units'],
            shift = info['shift'],
            scale = info['scale'],
            cmap  = info['cmap'],
            dcmap = info['dcmap'],
            axes  = axes,
            ax0   = i*5
        )
    plt.tight_layout()
    plt.savefig('figs/f_cf_era5' + region_info['dir'] + '.png', dpi = 300)


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
    try:
        cube = iris.load_cube(file)
    except:
        cube = iris.load(file)
        cube = [cb for cb in cube if len(cb.shape) == 3][0]
        
    cube0 = cube.copy()
    season = sub_year_months(cube, months)
    try:
        season_year = sub_year_range(season, years)
    except:
        set_trace()
    if len(months) > 1:
        season_year = season_year.collapsed('time', iris.analysis.MEAN)
    season_year.data[season_year.data > 9E9] = np.nan
    return season_year    

def plot_factual_and_cf(variable, factual_path, cf_dir, months, years, label=None, units = '',
                        shift = 0.0, scale = 1.0, vrange = None,
                        cmap = 'gradient_hues', dcmap = 'diverging_TealOrange',
                        axes = None, ax0 = None, eg_cube = None):
    # Load factual
    factual_avg = load_and_average_months(factual_path, months, years) + shift
    if eg_cube is None:
        eg_cube = factual_avg
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
    
    if vrange is None:
        extend = 'both'
    elif vrange[0] is None:
        levels = np.append(levels, vrange[1])
        extend = 'min'
    elif vrange[1] is None:
        levels = np.append(vrange[0], levels)
        extend = 'max'
    else:
        levels = np.append(np.append(vrange[0], levels), vrange[1])
        extend = 'neither'
    levels = np.unique(levels)
    
    if axes is None:
        fig, axes = set_up_sow_plot_windows(5, 1, factual_avg)
        ax0 = 0
    cbar_label = variable + ' (' + units + ')'

    def plt_mask(cube, title, cmap, cbar_label, levels, axi = 0, extend = 'both'):
        cube.data[np.isnan(eg_cube.data)] = np.nan
        cube.data.mask = eg_cube.data.mask
        if extend == 'max':
            cube.data[cube.data == 0] = 0.00000001
        
        plot_map_sow(cube, title=title, cmap = cmap,
                 cbar_label=cbar_label, levels = levels, ax = axes[ax0 + axi], extend = extend)
    #set_trace()
    plt_mask(factual_avg, f"Factual {title_label}", cmap, variable, levels, 0, extend)
    plt_mask(cf_10, f"CF 10th percentile {title_label}", cmap, cbar_label, levels, 1, extend)
    plt_mask(cf_90, f"CF 90th percentile {title_label}", cmap, cbar_label, levels, 2, extend)
    levels = auto_pretty_levels([diff_10, diff_90], 5)
    plt_mask(diff_10, f"Difference (Factual - CF 10th percentile) {title_label}", 
             dcmap, f"Δ {cbar_label}", levels, 3)
    plt_mask(diff_90, "Difference (Factual - CF 90th percentile) {title_label}", 
             dcmap, f"Δ {cbar_label}", levels,4)

    return eg_cube

variable_info = {'tas_mean':{"file": 'tas_mean', 'label': 'Mean Monthly Temp', 'Units': "°C", 
                            'shift': -273.15, 'scale': 1.0,
                            'range': None, 
                            'cmap': SoW_cmap['gradient_red'], 
                            'dcmap': SoW_cmap['diverging_BlueRed']},
                 'tas_max':{"file": 'tas_max', 'label': 'Max Monthly Temp', 'Units': "°C", 
                            'shift': -273.15, 'scale': 1.0, 
                            'range': None,
                            'cmap': SoW_cmap['gradient_red'], 
                            'dcmap': SoW_cmap['diverging_BlueRed']},
                 'precip': {"file": 'precip', 'label': 'Precipitation', 'Units': "mm/day",
                            "shift": 0.0, 'scale': 1.0, 
                            'range': [0.0, None],
                            'cmap': SoW_cmap['gradient_teal'], 
                            'dcmap': SoW_cmap['diverging_TealOrange'].reversed()},
                 'dry_days': {"file": 'dry_days', 'label': 'Mean. no dry days', 
                              'Units': "fraction",
                              "shift": 0.0, 'scale': 1.0, 
                            'range': [0.0, None],
                              'cmap': SoW_cmap['gradient_teal'].reversed(), 
                              'dcmap': SoW_cmap['diverging_TealOrange']},
                 'max_consec_dry': {"file": 'dry_days', 'label': 'Max. no consecutive dry days',
                              'Units': "no. days",
                              "shift": 0.0, 'scale': 1.0, 
                            'range': [0.0, None],
                              'cmap': SoW_cmap['gradient_teal'].reversed(), 
                              'dcmap': SoW_cmap['diverging_TealOrange']},
                 'hurs_mean': {"file": 'hurs_mean', 'label': 'Humidity', 
                               'Units': "%",
                               "shift": 0.0, 'scale': 1.0, 
                            'range': [0.0, 100.0],
                               'cmap': SoW_cmap['gradient_hotpink'], 
                               'dcmap': SoW_cmap['diverging_TealPurple'].reversed()},
                 'hurs_min': {"file": 'hurs_min', 'label': 'Min. Humidity', 
                               'Units': "%",
                               "shift": 0.0, 'scale': 1.0, 
                            'range': [0.0, 100.0],
                               'cmap': SoW_cmap['gradient_hotpink'], 
                               'dcmap': SoW_cmap['diverging_TealPurple'].reversed()},
                 'wind_max': {"file": 'wind_max', 'label': 'Max. Wind', 
                               'Units': "m/s",
                               "shift": 0.0, 'scale': 1.0, 
                            'range': [0.0, None],
                               'cmap': SoW_cmap['gradient_purple'], 
                               'dcmap': SoW_cmap['diverging_GreenPink'].reversed()}}

variables = ['tas_max', 'precip', 'dry_days', 'max_consec_dry', 'hurs_mean', 'hurs_min',
             'wind_max']

    
regions = ['LA', 'Congo', 'Amazon', 'Pantanal']

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
            vrange = info['range'],
            axes  = axes,
            ax0   = i*5, eg_cube = eg_cube[0],
        )
    plt.tight_layout()
    plt.savefig('figs/f_cf_era5' + region_info['dir'] + '.png', dpi = 300)


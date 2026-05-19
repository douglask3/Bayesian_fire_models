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
    
    if cube.ndim > 2:
        season = sub_year_months(cube, months)  
        season_year = sub_year_range(season, years)
        if len(months) > 1 and season_year.ndim > 2:
            season_year = season_year.collapsed('time', iris.analysis.MEAN)
        season_year.data[season_year.data > 9E9] = np.nan
        
        return season_year   
    else:
        return cube 

def plot_factual_and_cf(variable, factual_path, cf_dir, months, years, label=None, units = '',
                        shift = 0.0, scale = 1.0, vrange = None,
                        cmap = 'gradient_hues', dcmap = 'diverging_TealOrange',
                        axes = None, ax0 = None, eg_cube = None):
    
    # Load factual
    factual_avg = load_and_average_months(factual_path, months, years) + shift
    if eg_cube is None:
        eg_cube = factual_avg.copy()
    # Load counterfactual ensemble
    cf_cubes = []
    files  = glob.glob(cf_dir + '**', recursive = True)
    for file in files:
        print(file)
        if file.endswith(".nc") and variable in file:
            cube_avg = load_and_average_months(file, months, years) + shift
            cf_cubes.append(cube_avg)
            
    #set_trace()
    # Concatenate across a new dimension
    if not cf_cubes:
        Warning("No counterfactual files found for variable:", variable)
        
        levels = auto_pretty_levels(factual_avg.data.flatten(), ignore_v = shift)
    else:
        ref_cube = factual_avg.copy()
        cf_data = np.stack([cube.data for cube in cf_cubes], axis=0)
        
        # Compute 10th, 50th, 90th percentiles across realisations
        cf_10 = ref_cube.copy(data=np.percentile(cf_data, 10, axis=0))
        cf_10.data[factual_avg.data.mask] = np.nan
        cf_90 = ref_cube.copy(data=np.percentile(cf_data, 90, axis=0))
        cf_90.data[factual_avg.data.mask] = np.nan
        diff_10 = ref_cube.copy(data=factual_avg.data - cf_10.data)
        diff_90 = ref_cube.copy(data=factual_avg.data - cf_90.data)
         
        levels = auto_pretty_levels(np.append(np.append(cf_10.data.flatten(), 
                                cf_90.data.flatten()), factual_avg.data.flatten()), 
                                ignore_v = shift)
    
        # Plot
    title_label = label or variable
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
    
    plt_mask(factual_avg, f"Factual {title_label}", cmap, variable, levels, 0, extend)
    try:
        plt_mask(cf_10, f"CF 10th percentile {title_label}", cmap, cbar_label, 
                 levels, 1, extend)
        plt_mask(cf_90, f"CF 90th percentile {title_label}", cmap, cbar_label, 
                 levels, 2, extend)
        levels = auto_pretty_levels([diff_10, diff_90], 5)
        
        if levels[0] >= 0 or levels[-1] <= 0:
            dcmap = cmap
        plt_mask(diff_10, f"Difference (Factual - CF 10th percentile) {title_label}", 
                  dcmap, f"Δ {cbar_label}", levels, 3)
        plt_mask(diff_90, f"Difference (Factual - CF 90th percentile) {title_label}", 
                  dcmap, f"Δ {cbar_label}", levels,4)
    except:
        pass

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
                 'pr': {"file": 'pr', 'label': 'Precipitation', 'Units': "mm/day",
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
                 'cumm_dry_days_max': {"file": 'cumm_dry_days_max', 'label': 'Max. no consecutive dry days',
                              'Units': "no. days",
                              "shift": 0.0, 'scale': 1.0, 
                            'range': [0.0, None],
                              'cmap': SoW_cmap['gradient_teal'].reversed(), 
                              'dcmap': SoW_cmap['diverging_TealOrange']},
                 #'hurs_mean': {"file": 'hurs_mean', 'label': 'Humidity', 
                 #              'Units': "%",
                 #              "shift": 0.0, 'scale': 1.0, 
                 #           'range': [0.0, 100.0],
                 #              'cmap': SoW_cmap['gradient_hotpink'], 
                 #s              'dcmap': SoW_cmap['diverging_TealPurple'].reversed()},
                 'hursmin_mean': {"file": 'hursmin_mean', 'label': 'Min. Humidity', 
                               'Units': "%",
                               "shift": 0.0, 'scale': 1.0, 
                            'range': [0.0, 1.0],
                               'cmap': SoW_cmap['gradient_hotpink'], 
                               'dcmap': SoW_cmap['diverging_TealPurple'].reversed()},
                 'gust1_max': {"file": 'gust1_max', 'label': 'Max. Gust Wind', 
                               'Units': "m/s",
                               "shift": 0.0, 'scale': 1.0, 
                            'range': [0.0, None],
                               'cmap': SoW_cmap['gradient_purple'], 
                               'dcmap': SoW_cmap['diverging_TealPurple'].reversed()},
                 'gust1_mean': {"file": 'gust1_mean', 'label': 'Mean Gust Wind', 
                               'Units': "m/s",
                               "shift": 0.0, 'scale': 1.0, 
                            'range': [0.0, None],
                               'cmap': SoW_cmap['gradient_purple'], 
                               'dcmap': SoW_cmap['diverging_TealPurple'].reversed()},
                 'wind_max': {"file": 'wind_max', 'label': 'Max. Wind', 
                               'Units': "m/s",
                               "shift": 0.0, 'scale': 1.0, 
                            'range': [0.0, None],
                               'cmap': SoW_cmap['gradient_purple'], 
                               'dcmap': SoW_cmap['diverging_GreenPink'].reversed()},
                 'Wood': {"file": 'wood_HYDE31', 'label': 'Woody cover', 
                               'Units': "",
                               "shift": 0.0, 'scale': 1.0, 
                               'range': [0.0, 1.0],
                               'cmap': SoW_cmap['gradient_teal'], 
                               'dcmap': SoW_cmap['diverging_GreenPink'].reversed()},
                 'Tree': {"file": 'tree_HYDE31', 'label': 'Tree cover', 
                               'Units': "",
                               "shift": 0.0, 'scale': 1.0, 
                               'range': [0.0, 1.0],
                               'cmap': SoW_cmap['gradient_teal'], 
                               'dcmap': SoW_cmap['diverging_GreenPink'].reversed()},
                 'Veg_cover': {"file": 'veg_HYDE31', 'label': 'Veg cover', 
                               'Units': "",
                               "shift": 0.0, 'scale': 1.0, 
                               'range': [0.0, 1.0],
                               'cmap': SoW_cmap['gradient_teal'], 
                               'dcmap': SoW_cmap['diverging_GreenPink'].reversed()},
                 'Veg_cover_log': {"file": 'veg_HYDE31_log', 'label': 'Log of Veg cover', 
                               'Units': "",
                               "shift": 0.0, 'scale': 1.0, 
                               'range': [0.0, 1.0],
                               'cmap': SoW_cmap['gradient_teal'], 
                               'dcmap': SoW_cmap['diverging_GreenPink'].reversed()},
                 'Cropland': {"file": 'cropland_regridded_to_era5', 'label': 'Cropland', 
                               'Units': "",
                               "shift": 0.0, 'scale': 1.0, 
                               'range': [0.0, 1.0],
                               'cmap': SoW_cmap['gradient_greys'], 
                               'dcmap': SoW_cmap['diverging_GreenPink'].reversed()},
                 'Pasture': {"file": 'pasture_regridded_to_era5', 'label': 'Pasture', 
                               'Units': "",
                               "shift": 0.0, 'scale': 1.0, 
                               'range': [0.0, 1.0],
                               'cmap': SoW_cmap['gradient_greys'], 
                               'dcmap': SoW_cmap['diverging_GreenPink'].reversed()},
                 'Burned Area': {"file": 'burned_area', 'label': 'Burned Area', 
                               'Units': "",
                               "shift": 0.0, 'scale': 1.0, 
                               'range': [0.0, 1.0],
                               'cmap': SoW_cmap['gradient_red'], 
                               'dcmap': SoW_cmap['diverging_TealOrange'].reversed()}}

variables = ["Veg_cover_log", 'Burned Area', 'tas_mean', 'tas_max',  'gust1_mean', 'Cropland', 'Pasture', 'Veg_cover', 'Tree', 'Wood', 'pr', 'dry_days', 'cumm_dry_days_max', 'hursmin_mean', 'gust1_mean']

    
regions = [ 'Northwest_Iberia', 'Chilean_Temperate_Forests_and_Matorral', 'Midwestern_Canadian_Shield_forests','Scottish_Highlands', 'Southeast_South_Korea']

for region in regions:
    #region_info = get_region_info(region)[region]
    
    eg_cube = iris.load_cube('data/data/driving_data2526/' + region + \
                           '/nrt/factual/' + variable_info[variables[0]]['file'] + '.nc')
    fig, axes = set_up_sow_plot_windows(len(variables), 5, 
                                        eg_cube = eg_cube, figsize = (30, 30))
    if eg_cube.ndim == 3:
        eg_cube = eg_cube[0]
    for i, variable in enumerate(variables):
        info = variable_info[variable]
        plot_factual_and_cf(
            variable =  info['file'],
            factual_path = 'data/data/driving_data2526/' + region + \
                           '/nrt/factual/' + info['file'] + '.nc',
            cf_dir = 'data/data/driving_data2526/' + region + \
                     '/nrt//countfactual//',
            months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
            years = [2025],
            label = info['label'],
            units = info['Units'],
            shift = info['shift'],
            scale = info['scale'],
            cmap  = info['cmap'],
            dcmap = info['dcmap'],
            vrange = info['range'],
            axes  = axes,
            ax0   = i*5, eg_cube =eg_cube
        )
    plt.tight_layout()
    plt.savefig('figs/f_cf_era5' + region + '.png', dpi = 300)


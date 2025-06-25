import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import glob
import os
import matplotlib.colors as mcolors

import sys
sys.path.append('SoW_info/')
from state_of_wildfires_colours  import SoW_cmap
from state_of_wildfires_region_info  import get_region_info
sys.path.append('libs/')
from plot_maps import *


# Function to load and concatenate all files in a directory
def load_ensemble(base_dir, run_dir, exp):
    if isinstance(run_dir, list):
        out = load_ensemble(base_dir, run_dir[0], exp)
        out.data -= load_ensemble(base_dir, run_dir[1], exp).data
        return out
    directory = os.path.join(base_dir + '/' + run_dir + '/', exp)
    file_paths = sorted(glob.glob(os.path.join(directory, "sample-pred*.nc")))
    skips = max([1, round(len(file_paths)/max_smaples)])
    file_paths = file_paths[::skips]
    
    cubes = iris.cube.CubeList([iris.load_cube(fp) for fp in file_paths])
    return cubes.merge_cube()  # Merge along realization dimension if possible

# Prepare figure

def round_to_nice(v):
    """Round to nearest 'nice' number (1, 2, or 5 × 10^n)"""
    if v == 0:
        return 0
    exponent = np.floor(np.log10(abs(v)))
    fraction = abs(v) / 10**exponent
    if fraction < 1.5:
        nice = 1
    elif fraction < 3.5:
        nice = 2
    elif fraction < 7.5:
        nice = 5
    else:
        nice = 10
    return np.sign(v) * nice * 10**exponent

def get_nice_percentile_levels(cube, percentiles=[0,10,20,30,40,50,60,70,80,90]):
    data_flat = cube.data.compressed() if np.ma.is_masked(cube.data) else cube.data.flatten()
    raw_percentiles = np.percentile(data_flat[data_flat != 0], percentiles)
    
    nice_levels = [round_to_nice(v) for v in raw_percentiles]
    return sorted(set(nice_levels))  # remove duplicates

def plot_map(cube, title, cmap, levels, ax):
    plot_map_sow(cube, title, cmap=SoW_cmap[cmap], 
                    levels=levels,# extend = "neither",
                    ax=ax, cbar_label = "", ignore_v = 0)

from pdb import set_trace
# Define the base directory
base_dir = "outputs/outputs_scratch/ConFLAME_nrt-attribution9/Amazon-2425//samples/_19-frac_points_0.5//"
run_dirs = ["factual-", "counterfactual-"] 
# List of experiment subdirectories
experiments = ["control", "Evaluate", "Standard_0", "Standard_1", "Standard_2", "Standard_3" , "Standard_4"]

same_levels = False
max_smaples = 100

def plot_controls_for_dir(run_dir, 
                          cmaps = ['gradient_teal', 'gradient_hotpink', 'gradient_red'],
                          same_levels = False):
    for i, exp in enumerate(experiments):
        print(exp)
        cube = load_ensemble(base_dir, run_dir, exp)*100.0
        if i == 0:
            fig, axes = set_up_sow_plot_windows(7, 3, cube, size_scale = 6, flatten = False)
        # Compute 10th and 90th percentiles
        annual_average = cube.collapsed('time', iris.analysis.MEAN)
        p10 = annual_average.collapsed('realization', iris.analysis.PERCENTILE, percent=10)
        p90 = annual_average.collapsed('realization', iris.analysis.PERCENTILE, percent=90)
        mean= annual_average.collapsed('realization', iris.analysis.MEAN)
        p10.rename(f"{exp} - 10th %ile")
        mean.rename(f"{exp} - mean")
        p90.rename(f"{exp} - 90th %ile")
        
        def define_level(cbs):
            return auto_pretty_levels(cbs, n_levels = 7, ignore_v = 0.0)
            
        if same_levels:        
            levels =  define_level([p10, p90])
        else:
            levels = 'auto'
        
        plot_map(p10 , f"{exp} - 10th %ile", cmaps[0], levels, axes[i, 0])   
        plot_map(mean, f"{exp} - mean", cmaps[1], levels, axes[i, 1])  
        plot_map(p90 , f"{exp} - 90th %ile", cmaps[2], levels, axes[i, 2])
    
    plt.tight_layout()
    if isinstance(run_dir, list):
        run_dir = run_dir[1] + run_dir[0]
    fig.savefig(base_dir + "controls" + run_dir + ".png", dpi=300, bbox_inches="tight")


plot_controls_for_dir(run_dirs[0])
plot_controls_for_dir(run_dirs[1])
plot_controls_for_dir(run_dirs, 
                      cmaps = ['diverging_TealOrange'] * 3, same_levels = True)

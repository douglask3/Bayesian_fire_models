import iris
import numpy as np
import glob

import sys
sys.path.append('SoW_info/')
from state_of_wildfires_colours  import SoW_cmap
from state_of_wildfires_region_info  import get_region_info
sys.path.append('libs/')
from plot_maps import *

from pdb import set_trace

experiments = ["control", "Standard_0", "Standard_1", "Standard_2"]
# Path
path = "outputs/outputs/model_test_baseline_1/samples/_9-frac_points_0.001/baseline-/"

def plot_trend(p50_cube, prob_cube, name, cmap, add_cbar, levels, axes, axi):
    
    if levels is None: 
        levels = auto_pretty_levels(p50_cube.data, n_levels=3)
    plot_map_sow(p50_cube, name, cmap = cmap, levels = levels, ax = axes[axi * 2], 
                 add_cbar = add_cbar)
    if axi == 0:
        axes[0].text(
            0, 0.5, "Trend",
            transform=axes[0].transAxes,
            rotation=90,
            va='center',
            ha='right'
        )
    plot_map_sow(prob_cube, ax = axes[axi * 2 + 1], extend = 'neither', 
                 levels = [0, 0.01, 0.05, 0.1, 0.9, 0.95, 0.99, 1], add_cbar = add_cbar)
    if axi == 0:
        axes[1].text(
            0, 0.5, "Likelihood > 0",
            transform=axes[1].transAxes,
            rotation=90,
            va='center',
            ha='right'
        )
    return levels


def plot_experiment(experiment, axi, cmap, levels, name, fig = None, axes  = None):
    files = sorted(glob.glob(path + experiment + "/sample-pred*.nc"))
    
    diffs = []
    
    for f in files:
        cube = iris.load_cube(f)
        
        data = cube.data  # shape: (time, lat, lon)
        
        # Split into halves
        first_half = data[:12, :, :]
        second_half = data[12:, :, :]
        
        # Mean over time
        first_mean = np.mean(first_half, axis=0)
        second_mean = np.mean(second_half, axis=0)
        
        # Difference
        diff = second_mean - first_mean
        
        diffs.append(diff)
    
    # Stack into (n_samples, lat, lon)
    diffs = np.stack(diffs, axis=0) * 10
    
    p5  = np.percentile(diffs, 5, axis=0)
    p50 = np.percentile(diffs, 50, axis=0)
    p95 = np.percentile(diffs, 95, axis=0)
    
    prob_gt0 = np.mean(diffs > 0, axis=0) #* 100  # %
    prob_is0 = np.mean(diffs == 0, axis=0)
    prob_gt0 = prob_gt0 + 0.5*prob_is0
    prob_gt0[np.isnan(p50)] = np.nan
    
    template = iris.load_cube(files[0])[0, :, :]  # just lat/lon grid
    
    def make_cube(data, name):
        cube = template.copy(data=data)
        cube.rename(name)
        return cube

    p5_cube  = make_cube(p5,  "delta_p5")
    p50_cube = make_cube(p50, "delta_p50")
    p95_cube = make_cube(p95, "delta_p95")
    prob_cube = make_cube(prob_gt0, "prob_delta_gt0")

    if axes is None:
        fig, axes = set_up_sow_plot_windows(2, len( experiments), p50_cube, transpose = True)   
    

    if axi == len(experiments) -1:
        add_cbar = True
    else:
        add_cbar = False
    
    levels = plot_trend(p50_cube, prob_cube, name, cmap, add_cbar, levels, axes, axi)
    
    return fig, axes, levels

cmap = SoW_cmap['diverging_GreenPink']
cmap = cmap.reversed()
#set_trace()
axes = None
fig = None
levels = None

names = ['Model fapar', 'climate +', 'climate -', 'soil']
for i, experiment in enumerate(experiments):
    fig, axes, levels = plot_experiment(experiment, i, cmap, levels, names[i], fig, axes)    
    
set_trace()

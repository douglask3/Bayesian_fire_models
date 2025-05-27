import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import glob
import os
import matplotlib.colors as mcolors
from pdb import set_trace
# Define the base directory
base_dir = "outputs/outputs/ConFire_nrt4Amazon-2425-fuel2/samples/_19-frac_points_0.5//baseline-/"

# List of experiment subdirectories
experiments = ["control", "Evaluate", "Standard_0", "Standard_1", "Standard_2", "Standard_3" , "Standard_4"] #, "Potential0", "Potential1"

same_norm = False

max_smaples = 100

# Function to load and concatenate all files in a directory
def load_ensemble(directory):
    file_paths = sorted(glob.glob(os.path.join(directory, "sample-pred*.nc")))
    skips = max([1, round(len(file_paths)/max_smaples)])
    file_paths = file_paths[::skips]
    
    cubes = iris.cube.CubeList([iris.load_cube(fp) for fp in file_paths])
    return cubes.merge_cube()  # Merge along realization dimension if possible

# Prepare figure
fig, axes = plt.subplots(len(experiments), 3, figsize=(15, 25), subplot_kw={'projection': ccrs.PlateCarree()})

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

def plot_map(cube, norm = None, *args, **kw):
    #if norm is None:
    #    norm = mcolors.LogNorm(vmin=np.max([cube.data.max()/1000000000, cube.data.min()]), 
    #                           vmax=cube.data.max())
    #set_trace()
    #levels = get_nice_percentile_levels(cube)
    #print(levels)
    #if len(levels) == 1: set_trace()
    
    #set_trace()
    qplt.contourf(cube, norm=norm, *args, **kw)


for i, exp in enumerate(experiments):
    print(exp)
    cube = load_ensemble(os.path.join(base_dir, exp))
     
    # Compute 10th and 90th percentiles
    annual_average = cube.collapsed('time', iris.analysis.MEAN)
    p10 = annual_average.collapsed('realization', iris.analysis.PERCENTILE, percent=10)
    p90 = annual_average.collapsed('realization', iris.analysis.PERCENTILE, percent=90)
    mean= annual_average.collapsed('realization', iris.analysis.MEAN)
    p10.rename(f"{exp} - 10th %ile")
    mean.rename(f"{exp} - mean")
    p90.rename(f"{exp} - 90th %ile")
    
    if same_norm:        
        norm = mcolors.LogNorm(vmin=np.max([0.00000000000001, p10.data.min()]), vmax=p90.data.max())
    else:
        norm = None
    # Plot 10th percentile
    ax = axes[i, 0]
    ax.set_title(f"{exp} - 10th %ile")
    plot_map(p10, norm=norm, axes=ax, cmap='Blues')

    ax.set_title(f"{exp} - mean")
    ax = axes[i, 1]
    plot_map(mean, norm=norm, axes=ax, cmap='Greens')

    # Plot 90th percentile
    ax = axes[i, 2]
    ax.set_title(f"{exp} - 90th %ile")
    plot_map(p90, norm=norm, axes=ax, cmap='Reds')
    
plt.tight_layout()
fig.savefig(base_dir + "controls.png", dpi=300, bbox_inches="tight")


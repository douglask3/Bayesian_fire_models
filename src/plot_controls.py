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
base_dir = "outputs/outputs/ConFire_AR7-correctedpower-masked6-clog3-Modreturn/samples/_6-frac_points_0.2/baseline-/"

# List of experiment subdirectories
experiments = ["control", "Evaluate", "Standard_0", "Standard_1"] #, "Potential0", "Potential1"

# Function to load and concatenate all files in a directory
def load_ensemble(directory):
    file_paths = sorted(glob.glob(os.path.join(directory, "sample-pred*.nc")))
    cubes = iris.cube.CubeList([iris.load_cube(fp) for fp in file_paths])
    return cubes.merge_cube()  # Merge along realization dimension if possible

# Prepare figure
fig, axes = plt.subplots(2, len(experiments), figsize=(15, 6), subplot_kw={'projection': ccrs.PlateCarree()})

for i, exp in enumerate(experiments):
    print(exp)
    cube = load_ensemble(os.path.join(base_dir, exp))
     
    # Compute 10th and 90th percentiles
    annual_average = cube.collapsed('time', iris.analysis.MEAN)
    p10 = annual_average.collapsed('realization', iris.analysis.PERCENTILE, percent=10)
    p90 = annual_average.collapsed('realization', iris.analysis.PERCENTILE, percent=90)
    p10.rename(f"{exp} - 10th %ile")
    p90.rename(f"{exp} - 90th %ile")

    norm = mcolors.LogNorm(vmin=np.max([0.000001, p10.data.min()]), vmax=p90.data.max())
    
    # Plot 10th percentile
    ax = axes[0, i]
    ax.set_title(f"{exp} - 10th %ile")
    qplt.contourf(p10, axes=ax, cmap='Blues', norm=norm)

    # Plot 90th percentile
    ax = axes[1, i]
    ax.set_title(f"{exp} - 90th %ile")
    qplt.contourf(p90, axes=ax, cmap='Reds', norm=norm)

plt.tight_layout()
fig.savefig(base_dir + "controls.png", dpi=300, bbox_inches="tight")


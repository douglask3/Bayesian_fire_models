import os
import glob
import numpy as np
import iris
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.animation as animation
import imageio.v2 as imageio  # for GIF assembly

import sys
sys.path.append('SoW_info/')
from state_of_wildfires_colours  import SoW_cmap
from state_of_wildfires_region_info  import get_region_info

sys.path.append('libs/')
from plot_maps import *

# --- Directories ---
base_dir = "outputs/outputs_scratch/ConFLAME_nrt-attribution-base-2/Pantanal/samples/_20-frac_points_0.2"
factual_dir = os.path.join(base_dir, "factual-/control")
counter_dir = os.path.join(base_dir, "counterfactual-/control")

# --- Output directory ---
os.makedirs("figs", exist_ok=True)
gif_out = "figs/pantanal_burned_area_comparison.gif"

outpath = "figs/cop_maps/"
os.makedirs(outpath, exist_ok=True)

# --- File lists (sorted to match pairs) ---
factual_files = sorted(glob.glob(os.path.join(factual_dir, "sample-pred*.nc")))
counter_files = sorted(glob.glob(os.path.join(counter_dir, "sample-pred*.nc")))

# Limit to ~20 pairs
n_pairs = min(100, len(factual_files))
factual_files = factual_files[:n_pairs]
counter_files = counter_files[:n_pairs]

# --- Load one sample to define coords ---
sample_cube = iris.load_cube(factual_files[0])
lons = sample_cube.coord('longitude').points
lats = sample_cube.coord('latitude').points

# --- Preload all mean maps ---
factual_means = []
counter_means = []
diff_means = []
from pdb import set_trace
import iris.quickplot as qplt
import matplotlib.pyplot as plt
for f_file, c_file in zip(factual_files, counter_files):
    f_cube = iris.load_cube(f_file)
    c_cube = iris.load_cube(c_file)
    
    f_mean = f_cube.collapsed('time', iris.analysis.MEAN)*100
    c_mean = c_cube.collapsed('time', iris.analysis.MEAN)*100
    
    diff = f_mean - c_mean
    
    if len(diff_means) == 0:
        diff_means_mean = diff / n_pairs
    else:
        diff_means_mean.data =+ diff.data / n_pairs
    
    factual_means.append(f_mean)
    counter_means.append(c_mean)
    
    diff_means.append(diff)


fig, axes = set_up_sow_plot_windows(1,1, factual_means[0], size_scale = 6)
levels = auto_pretty_levels(diff_means_mean.data.flatten(), n_levels=4, force0= True)
plot_map_sow(diff_means_mean, 'Average difference caused climate change', ax = axes, levels = levels)
plt.savefig('figs/Amazonia_mean_diff.png', dpi=150, bbox_inches='tight')

set_trace()
def update(i):
    
    # --- Plot setup ---
    fig, axes = set_up_sow_plot_windows(2, 3, factual_means[0], size_scale = 6)
    
    levels = auto_pretty_levels(np.append(factual_means[i].data.flatten(), 
                                          counter_means[i].data.flatten()),  
                                          n_levels=7, ignore_v = 0.0)
    if levels[-1] < 0.0001:
        factual_means[i].data = factual_means[i].data * 100
        counter_means[i].data = counter_means[i].data * 100
        diff_means[i].data = diff_means[i].data * 100
        levels = levels * 100
    
    im_f = plot_map_sow(factual_means[i], 'With climate change', levels = np.append(0, levels), 
                 cmap=SoW_cmap['gradient_red'], extend = 'max', ax = axes[0])
    im_c = plot_map_sow(counter_means[i], 'Without climate change', 
                        levels = np.append(0, levels), 
                 cmap=SoW_cmap['gradient_red'], extend = 'max', ax = axes[1])
    levels = auto_pretty_levels(diff_means[i].data.flatten(), n_levels=4, force0= True)
    im_d = plot_map_sow(diff_means[i], 'Difference caused climate change',
                        ax = axes[2], levels = levels)
    
    
#    im_f.set_array(factual_means[i].ravel())
#    im_c.set_array(counter_means[i].ravel())
#    im_d.set_array(diff_means[i].ravel())
#    fig.suptitle(f"Sample {i+1}", fontsize=14)
#    return [im_f, im_c, im_d]

    fname = os.path.join(outpath, f"frame_{i:03d}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return fname
png_files = [update(i) for i in range(n_pairs)]

gif_path = os.path.join(outpath, "pantanal_burned_area_comparison.gif")
with imageio.get_writer(gif_path, mode='I', fps=3, loop=0) as writer:
    for fname in png_files:
        writer.append_data(imageio.imread(fname))
## --- Save GIF ---
#ani.save(gif_out, writer="pillow", fps=1)
#plt.close()

print(f"Saved animation: {gif_path}")


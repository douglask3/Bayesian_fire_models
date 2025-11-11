import os
import glob
import numpy as np
import iris
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.animation as animation

# --- Directories ---
base_dir = "outputs/outputs_scratch/ConFLAME_nrt-attribution-base-2/Amazon/samples/_19-frac_points_0.2"
factual_dir = os.path.join(base_dir, "factual-/control")
counter_dir = os.path.join(base_dir, "counterfactual-/control")

# --- Output directory ---
os.makedirs("figs", exist_ok=True)
gif_out = "figs/amazon_burned_area_comparison.gif"

# --- File lists (sorted to match pairs) ---
factual_files = sorted(glob.glob(os.path.join(factual_dir, "sample-pred*.nc")))
counter_files = sorted(glob.glob(os.path.join(counter_dir, "sample-pred*.nc")))

# Limit to ~20 pairs
n_pairs = min(20, len(factual_files))
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

for f_file, c_file in zip(factual_files, counter_files):
    f_cube = iris.load_cube(f_file)
    c_cube = iris.load_cube(c_file)
    
    f_mean = f_cube.collapsed('time', iris.analysis.MEAN).data
    c_mean = c_cube.collapsed('time', iris.analysis.MEAN).data
    
    diff = f_mean - c_mean
    
    factual_means.append(f_mean)
    counter_means.append(c_mean)
    diff_means.append(diff)

# Convert to numpy arrays
factual_means = np.array(factual_means)
counter_means = np.array(counter_means)
diff_means = np.array(diff_means)

# --- Color scales ---
vmax = np.percentile(np.concatenate([factual_means, counter_means]), 99)
diff_absmax = np.max(np.abs(diff_means))

# --- Plot setup ---
proj = ccrs.PlateCarree()
fig, axes = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection': proj})
titles = ["Factual", "Counterfactual", "Difference (F−C)"]

# Create meshgrid for pcolormesh
lon2d, lat2d = np.meshgrid(lons, lats)

# Initialize plots
im_f = axes[0].pcolormesh(lon2d, lat2d, factual_means[0], cmap="YlOrRd", vmin=0, vmax=vmax, transform=proj)
im_c = axes[1].pcolormesh(lon2d, lat2d, counter_means[0], cmap="YlOrRd", vmin=0, vmax=vmax, transform=proj)
im_d = axes[2].pcolormesh(lon2d, lat2d, diff_means[0], cmap="RdBu_r", vmin=-diff_absmax, vmax=diff_absmax, transform=proj)

for ax, t in zip(axes, titles):
    ax.coastlines(linewidth=0.5)
    ax.set_title(t, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

# Shared colorbars
fig.colorbar(im_f, ax=axes[0:2], orientation='horizontal', fraction=0.05, pad=0.08, label='Mean burned area')
fig.colorbar(im_d, ax=axes[2], orientation='horizontal', fraction=0.05, pad=0.08, label='Difference')

plt.tight_layout()

# --- Animation update function ---
def update(i):
    im_f.set_array(factual_means[i].ravel())
    im_c.set_array(counter_means[i].ravel())
    im_d.set_array(diff_means[i].ravel())
    fig.suptitle(f"Sample {i+1}", fontsize=14)
    return [im_f, im_c, im_d]

ani = animation.FuncAnimation(fig, update, frames=n_pairs, interval=800, blit=False)

# --- Save GIF ---
ani.save(gif_out, writer="pillow", fps=1)
plt.close()

print(f"Saved animation: {gif_out}")


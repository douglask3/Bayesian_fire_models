from pdb import set_trace
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import seaborn as sns
import numpy as np
import pandas as pd
import os
from matplotlib.backends.backend_pdf import PdfPages
import sys
sys.path.append('libs/')
from constrain_cubes_standard import *

from plot_maps import *
# Define paths
base_dir = "data/data/driving_data2526/Global/isimp3a/obsclim/GSWP3-W5E5/period_2000_2019/"

# File names and labels
files_labels = {
    "burned_area.nc": "Burned Area",
    "consec_dry_mean.nc": "Consecutive dry days",
    "debiased_nonetree_cover_jules-es.nc": "None tree vegetation cover",
    "debiased_tree_cover_jules-es.nc": "Tree cover",
    "pr_mean.nc": "Precipitation",
    "tas_max.nc": "Max. Monthly temperature",
    "tas_mean.nc": "Mean Temperature",
    "dry_days.nc": "no. dry days",
    "vpd_max.nc": "Max. VPD",
    "vpd_mean.nc": "mean VPD",
    "lightning.nc": "lightning",
    "pasture_jules-es.nc": "Pasture cover",
    "crop_jules-es.nc": "Crop cover",
    "urban_jules-es.nc": "Urban Cover",
    "debiased_tree_cover_change_jules-es.nc": "Change in tree cover",
    "crop_change_jules-es.nc": "change in crop cover"
}

# Matching colormaps (using seaborn-compatible Brewer palettes)
colormaps = {
    "Burned Area": "Oranges",
    "Consecutive dry days": "YlOrBr",
    "None tree vegetation cover": "Greens",
    "Tree cover": "YlGn",
    "Precipitation": "Blues",
    "Max. Monthly temperature": "Reds",
    "Mean Temperature": "RdYlBu",
    "no. dry days": "YlOrRd",
    "Max. VPD": "PuRd",
    "mean VPD": "PuBu",
    "lightning": "BuPu",
    "Pasture cover": "Greys",
    "Crop cover": "YlGnBu",
    "Urban Cover": "Purples",
    "Change in tree cover": "BrBG",
    "change in crop cover": "RdBu"
}

def plot_map(ax, data, title, cmap):#, vmin=None, vmax=None):
    #im = data.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(),
    #               cbar_kwargs={'shrink': 0.6, 'label': title})

    data = natural_earth_ocean_mask(data)
    plot_map_sow(data, title, 
                 add_cbar = True, extend = 'max',
                 cmap=cmap, use_pcolmesh = True, ax = ax,
                 cbar_orientation = 'horizontal', 
                 cbar_lab_rotate = 45, cbar_top_and_bottom = True)

def save_to_files(fig, fig_title, pdf):
    
    fig.savefig(f"{fig_title}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{fig_title}.pdf", bbox_inches='tight')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)  


import iris
def plot_isimip_drivers_for_region(pdf, region = 'Global'):
    print(region)
    base_dir = base_dir0 + region + "/isimp3a/obsclim/GSWP3-W5E5/period_2000_2019/"
    # Load data
    datasets = {}
    for file, label in files_labels.items():
        try:
            datasets[label] = iris.load_cube(os.path.join(base_dir, file))
        except: 
            pass
    # Identify land cover datasets (annual timestep)
    land_cover_labels = [label for label in datasets \
                         if any(x in label.lower() for x in ["cover", "change"])]
    

    # Grab any data array to get extent — ideally one of the variables
    
    example_data = next(iter(datasets.values()))
    lat = example_data.coord('latitude').points
    lon = example_data.coord('longitude').points

    lat_extent = lat.max() - lat.min()
    lon_extent = lon.max() - lon.min()

    # Tune this factor to your liking — 0.25 gives ~1 inch per 4 degrees
    scaling = 0.25
    map_width = lon_extent * scaling
    map_height = lat_extent * scaling

    # Account for number of maps (e.g., 4x4 grid)
    rows = int(np.ceil(np.sqrt(len(datasets))))
    cols = int(np.ceil(len(datasets)/rows))
    fig_width = map_width * cols
    fig_height = map_height * rows

    # --- 1. JFM Mean Maps or Annual Mean ---
    fig1, axes1 = set_up_sow_plot_windows(rows, cols, datasets['Burned Area'], size_scale = 3.0)
    #fig1, axes1 = plt.subplots(4, 4, figsize=(fig_width, fig_height), 
    #                           subplot_kw={'projection': ccrs.PlateCarree()})
    
    fig1.suptitle("Annual Mean (2000–2019)", fontsize=16)
    
    for i, (label, data) in enumerate(datasets.items()):
        plot_map(axes1[i], data.collapsed('time', iris.analysis.MEAN), label, colormaps[label])
    set_trace()

    # --- 2. Max Burned Area Month or Year ---
    
    burned = datasets["Burned Area"]
    jfm_burned = burned.where(is_jfm(burned), drop=True)
    burned_max_idx = jfm_burned.mean(dim=['lat', 'lon']).argmax().item()
    burned_max_time = jfm_burned['time'].isel(time=burned_max_idx).values
    burned_max_year = pd.to_datetime(str(burned_max_time)).year
    
    fig2, axes2 = plt.subplots(4, 4, figsize=(fig_width, fig_height), 
                               subplot_kw={'projection': ccrs.PlateCarree()})
    axes2 = axes2.flatten()
    fig2.suptitle(f"Values at Max Burned Area Month (JFM) – {np.datetime_as_string(burned_max_time, unit='M')}", fontsize=16)

    for i, (label, data) in enumerate(datasets.items()):
        if label in land_cover_labels:
            # Find year-matching index
            years = [pd.to_datetime(str(t)).year for t in data['time'].values]
            if burned_max_year in years:
                sel_index = years.index(burned_max_year)
                sel_data = data.isel(time=sel_index)
            else:
                sel_data = data.isel(time=0)  # fallback
        else:
            try:
                sel_data = data.sel(time=burned_max_time, method = 'nearest')
            except:
                set_trace()
        plot_map(axes2[i], sel_data, label, colormaps[label])

    # --- 3. Monthly Average Time Series ---
    fig3, axes3 = plt.subplots(4, 4, figsize=(18, 14))
    axes3 = axes3.flatten()
    fig3.suptitle("Monthly Average Time Series (2000–2019)", fontsize=16)
    
    for i, (label, data) in enumerate(datasets.items()):
        ts = data.mean(dim=["lat", "lon"])
        ts.plot(ax=axes3[i], color=sns.color_palette(colormaps[label])[4] \
                if label in colormaps else "black")
        axes3[i].set_title(label)
        axes3[i].set_xlabel("Time")
        axes3[i].set_ylabel("Mean Value")
    
    plt.tight_layout()

    fname = "figs/" + region + "_isimip_driving_"
    save_to_files(fig1, fname + "annual_average", pdf)
    save_to_files(fig2, fname + "at_max_BA", pdf)
    save_to_files(fig3, fname + "time_series", pdf)
    #plt.show()


base_dir0 = "data/data/driving_data2526/"

#regions = ['Amazon', 'NWIndia', 'Alberta', 'LA', 'Congo', 'Pantanal']
#
with PdfPages("fire_analysis_all_regions.pdf") as pdf:
    plot_isimip_drivers_for_region(pdf, 'Global')
#    for region in regions:
#        plot_isimip_drivers_for_region(pdf, region)



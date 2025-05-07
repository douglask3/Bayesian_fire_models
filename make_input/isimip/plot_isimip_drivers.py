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

# Define paths
base_dir = "data/data/driving_data2425/Amazon/isimp3a/obsclim/GSWP3-W5E5/period_2000_2019/"

# File names and labels
files_labels = {
    "burnt_area-2000-2019.nc": "Burned Area",
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

# Load data
datasets = {}
for file, label in files_labels.items():
    ds = xr.open_dataset(os.path.join(base_dir, file), decode_cf=True, mask_and_scale=True)    
    var_name = list(ds.data_vars)[0]
    data = ds[var_name]
    
    data.data[data.data>1e10] = np.nan
    datasets[label] = ds[var_name]

# Identify land cover datasets (annual timestep)
land_cover_labels = [label for label in datasets if any(x in label.lower() for x in ["cover", "change"])]

# Helper to extract JFM months
def is_jfm(ds):
    return ds['time.month'].isin([1, 2, 3])

def plot_map(ax, data, title, cmap):#, vmin=None, vmax=None):
    #im = data.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(),
    #               cbar_kwargs={'shrink': 0.6, 'label': title})
    data.plot(ax=ax, cmap=colormaps[label], transform=ccrs.PlateCarree(), cbar_kwargs={'shrink': 0.5})
    
    ax.coastlines()
    ax.set_title(title, fontsize=10)

    # Add Natural Earth features
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.5)

    # Admin level 1 (first-level subdivisions like states/provinces)
    admin1_shp = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none'
    )
    ax.add_feature(admin1_shp, edgecolor='gray', linewidth=0.4)
    ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.2)
    ax.add_feature(cfeature.OCEAN, facecolor='white')

def save_to_files(fig, fig_title):
    fig.savefig(f"{fig_title}.png", dpi=300, bbox_inches='tight')#, pad_inches=0.1)
    fig.savefig(f"{fig_title}.pdf", bbox_inches='tight')#, pad_inches=0.1)

# Grab any data array to get extent — ideally one of the variables
example_data = next(iter(datasets.values()))
lat = example_data['lat'].values
lon = example_data['lon'].values

lat_extent = lat.max() - lat.min()
lon_extent = lon.max() - lon.min()

# Tune this factor to your liking — 0.25 gives ~1 inch per 4 degrees
scaling = 0.25
map_width = lon_extent * scaling
map_height = lat_extent * scaling

# Account for number of maps (e.g., 4x4 grid)
cols = 4
rows = 4
fig_width = map_width * cols
fig_height = map_height * rows


# --- 1. JFM Mean Maps or Annual Mean ---
fig1, axes1 = plt.subplots(4, 4, figsize=(fig_width, fig_height), subplot_kw={'projection': ccrs.PlateCarree()})
axes1 = axes1.flatten()
fig1.suptitle("JFM or Annual Mean (2000–2019)", fontsize=16)

for i, (label, data) in enumerate(datasets.items()):
    if label in land_cover_labels:
        jfm_mean = data.mean("time")
    else:
        jfm_data = data.where(is_jfm(data), drop=True)
        jfm_mean = jfm_data.groupby("time.year").mean("time").mean("year")
    
    '''jfm_mean.plot(ax=axes1[i], cmap=colormaps[label], transform=ccrs.PlateCarree(), cbar_kwargs={'shrink': 0.5})
    axes1[i].set_title(label)
    axes1[i].coastlines( linewidth = 1.5)

    # Add Natural Earth features
    axes1[i].add_feature(cfeature.BORDERS, linewidth = 1.0)
    axes1[i].add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.67)
    '''
    plot_map(axes1[i], jfm_mean, label, colormaps[label])


# --- 2. Max Burned Area Month or Year ---
burned = datasets["Burned Area"]
jfm_burned = burned.where(is_jfm(burned), drop=True)
burned_max_idx = jfm_burned.mean(dim=['lat', 'lon']).argmax().item()
burned_max_time = jfm_burned['time'].isel(time=burned_max_idx).values
burned_max_year = pd.to_datetime(str(burned_max_time)).year

fig2, axes2 = plt.subplots(4, 4, figsize=(18, 14), subplot_kw={'projection': ccrs.PlateCarree()})
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
    '''sel_data.plot(ax=axes2[i], cmap=colormaps[label], transform=ccrs.PlateCarree(), cbar_kwargs={'shrink': 0.5})
    axes2[i].set_title(label)
    axes2[i].coastlines()
    '''
    plot_map(axes2[i], sel_data, label, colormaps[label])


# --- 3. Monthly Average Time Series ---
fig3, axes3 = plt.subplots(4, 4, figsize=(18, 14))
axes3 = axes3.flatten()
fig3.suptitle("Monthly Average Time Series (2000–2019)", fontsize=16)

for i, (label, data) in enumerate(datasets.items()):
    ts = data.mean(dim=["lat", "lon"])
    ts.plot(ax=axes3[i], color=sns.color_palette(colormaps[label])[4] if label in colormaps else "black")
    axes3[i].set_title(label)
    axes3[i].set_xlabel("Time")
    axes3[i].set_ylabel("Mean Value")

plt.tight_layout()

save_to_files(fig1, "figs/Amazon_isimip_driving_annaul_average")
save_to_files(fig2, "figs/Amazon_isimip_driving_at_max_BA")
save_to_files(fig3, "figs/Amazon_isimip_driving_time_series")

plt.show()

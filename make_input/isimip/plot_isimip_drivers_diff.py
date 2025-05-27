import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
import os
from pdb import set_trace

# Filepaths
base_dir = "data/data/driving_data2425/Amazon/isimp3a/"
scenarios = {
    "obsclim": f"{base_dir}/obsclim/GSWP3-W5E5/period_2000_2019/",
    "counterclim": f"{base_dir}/counterclim/GSWP3-W5E5/period_2000_2019/"
}

# Variables to compare (excluding burnt_area)
variables = [v for v in os.listdir(scenarios["obsclim"]) if v.endswith('.nc') and not v.startswith("burnt_area")]

# Plot settings
fig, axes = plt.subplots(4, 4, figsize=(18, 14), subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()

for i, varfile in enumerate(variables):
    print(f"Processing {varfile}")
    da_obs = xr.open_dataset(os.path.join(scenarios["obsclim"], varfile)).load()
    da_ctr = xr.open_dataset(os.path.join(scenarios["counterclim"], varfile)).load()

    var_name = list(da_obs.data_vars)[0]
    data_obs = da_obs[var_name]
    data_ctr = da_ctr[var_name]

    # Mask extreme values
    data_obs = data_obs.where(data_obs < 1e10)
    data_ctr = data_ctr.where(data_ctr < 1e10)

    # Monthly time selection (Jan–Mar)
    jan_mar_obs = data_obs.sel(time=data_obs['time.month'].isin([1, 2, 3]))
    jan_mar_ctr = data_ctr.sel(time=data_ctr['time.month'].isin([1, 2, 3]))

    # Annual means (e.g., group by year)
    try:
        annual_obs = jan_mar_obs.groupby('time.year').mean(dim='time')
        annual_ctr = jan_mar_ctr.groupby('time.year').mean(dim='time')
    except:
        set_trace()
    # Mean across all years
    mean_obs = annual_obs.mean(dim='year')
    mean_ctr = annual_ctr.mean(dim='year')

    # Difference (obsclim - counterclim)
    diff = mean_obs - mean_ctr

    # Plot
    ax = axes[i]
    im = diff.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='RdBu',
        norm=TwoSlopeNorm(vcenter=0),
        add_colorbar=False
    )
    ax.coastlines()
    ax.set_title(varfile.replace(".nc", ""))
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

plt.suptitle("Difference in Jan–Mar Annual Averages (obsclim - counterclim)", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.93)

# Save
plt.savefig("annual_diff_obsclim_vs_counterclim.png", dpi=300, bbox_inches='tight')
plt.savefig("annual_diff_obsclim_vs_counterclim.pdf", bbox_inches='tight')


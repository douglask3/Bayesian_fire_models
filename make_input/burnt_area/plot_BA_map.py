import iris
import iris.analysis
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.colors as mcolors

import sys
sys.path.append('../../libs/')
sys.path.append('libs/')
from constrain_cubes_standard import *

from pdb import set_trace
cube = iris.load_cube("data/data/driving_data2425/burnt_area0p05-uk.nc")
cube = constrain_natural_earth(cube, 'United Kingdom')

# Compute the **annual mean** over time
annual_mean_cube = cube.collapsed('time', iris.analysis.MEAN)

# Extract latitude, longitude, and data
lats = annual_mean_cube.coord('latitude').points
lons = annual_mean_cube.coord('longitude').points
burnt_area_data = annual_mean_cube.data

# Ensure burnt_area_data has shape (latitude, longitude)
if burnt_area_data.ndim == 3:  # (time, lat, lon)
    burnt_area_data = burnt_area_data[0]  # Take first (or mean over time)

# Set up figure with PlateCarree projection
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})

# Define colormap and **log scale** for burnt area
cmap = plt.cm.YlOrRd  # Yellow-Orange-Red for burnt area
norm = mcolors.LogNorm(vmin=np.nanpercentile(burnt_area_data[burnt_area_data > 0], 5),  # Avoid extreme outliers
                       vmax=np.nanpercentile(burnt_area_data[burnt_area_data > 0], 95))

# Plot data
img = ax.pcolormesh(lons, lats, burnt_area_data, cmap=cmap, norm=norm, shading='auto')

# Add colorbar
cbar = plt.colorbar(img, ax=ax, orientation='vertical', label='Annual Average Burnt Area (%)')

# Add country boundaries
ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')

# Add land and ocean colors
ax.add_feature(cfeature.LAND, color='white')  # Keep land visible
ax.add_feature(cfeature.OCEAN, color='cyan')  # Make ocean cyan

# **Add UK nation boundaries**
uk_nations = cfeature.NaturalEarthFeature(category='cultural',
                                          name='admin_0_map_units',
                                          scale='10m',
                                          facecolor='none')
ax.add_feature(uk_nations, linestyle='--', edgecolor='blue', linewidth=1)

# Set title
ax.set_title('Annual Average Burnt Area - UK')

# **Save as PDF**
plt.savefig("burnt_area_UK.pdf", format="pdf", bbox_inches="tight")

# Show plot
plt.show()      

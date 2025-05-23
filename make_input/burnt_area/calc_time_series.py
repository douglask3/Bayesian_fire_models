import iris
import numpy as np
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import datetime
import os
import pandas as pd

from pdb import set_trace

# List of regions
regions = ["Amazon", "Congo", "LA", "Pantanal"]
csv_dir_out = 'data/data/driving_data2425/'

# Set up a 2x3 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 8))#, constrained_layout=True)
axes = axes.flatten()  # Flatten to make it easier to loop through


for i, region in enumerate(regions):
    # Load the data
    cube = iris.load_cube("data/data/driving_data2425/" + region + "/burnt_area.nc")
    
    
    # Extract time coordinate
    time_coord = cube.coord('time') 
    time_points = [datetime.datetime(t.year, t.month, t.day) \
                   for t in time_coord.units.num2date(time_coord.points)]
    
    
    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube)
    mean_burnt_area = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, 
                                     weights=grid_areas).data.data
    q95 = cube.collapsed(['longitude', 'latitude'], iris.analysis.PERCENTILE, 
                         percent = [99]).data.data
    for tstep in range(cube.shape[0]): 
        cube.data.mask[tstep][cube.data[tstep] <= q95[tstep]] =  True
    grid_areas = iris.analysis.cartography.area_weights(cube)
    mean_extreme_burnt_area = cube.collapsed(['longitude', 'latitude'], 
                                             iris.analysis.MEAN, weights=grid_areas).data.data
    df = pd.DataFrame({
        'time': time_points,
        'mean_burnt_area': mean_burnt_area,
        'p95_burnt_area': mean_extreme_burnt_area
    })
    df.to_csv(csv_dir_out + region + '/burnt_area_data.csv', index=False)
    
    ax1 = axes[i]
    ax2 = ax1.twinx()

    ax2.plot(time_points, mean_extreme_burnt_area, 'r--', label='Extreme (95%)')
    ax1.plot(time_points, mean_burnt_area, 'b-', label='Mean')
    ax2.plot(time_points, mean_extreme_burnt_area, 'r--')

    ax1.set_title(region)
    #ax1.set_ylabel('Mean', color='b')
    #ax2.set_ylabel('Extreme', color='r')

    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')

# Tidy up the layout
fig.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.08, wspace=0.3, hspace=0.3)
#plt.tight_layout()
plt.suptitle("Burnt Area Time Series by Region", fontsize=16, y=1.02)
plt.subplots_adjust(top=0.88)  # Leave space for the main title
# Shared axis labels (like mtext in R)
fig.text(0.01, 0.5, 'Mean Burnt Area', va='center', ha='center', rotation='vertical', color='b', fontsize=12)
fig.text(0.985, 0.5, 'Extreme Burnt Area (≥95%)', va='center', ha='center', rotation='vertical', color='r', fontsize=12)

# Optional: Shared title
fig.suptitle("Burnt Area Time Series by Region", fontsize=16, y=1.03)

plt.show()

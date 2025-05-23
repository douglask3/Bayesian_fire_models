import iris
import numpy as np
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import datetime

from pdb import set_trace
# Load the data
cube = iris.load_cube("data/data/driving_data2425/Amazon/burnt_area.nc")


# Extract time coordinate
time_coord = cube.coord('time')
time_points = [datetime.datetime(t.year, t.month, t.day) for t in time_coord.units.num2date(time_coord.points)]


cube.coord('latitude').guess_bounds()
cube.coord('longitude').guess_bounds()
grid_areas = iris.analysis.cartography.area_weights(cube)
mean_burnt_area = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas).data.data
q95 = cube.collapsed(['longitude', 'latitude'], iris.analysis.PERCENTILE, percent = [99]).data.data
for i in range(cube.shape[0]): 
     #cube.data.data[i][cube.data[i] <= q95[i]] =  np.nan
     cube.data.mask[i][cube.data[i] <= q95[i]] =  True
grid_areas = iris.analysis.cartography.area_weights(cube)
mean_extreme_burnt_area = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas).data.data

'''# Loop through time slices
for i in range(cube.shape[0]):
    print(i)
    slice_i = cube[i]
    data = slice_i.data

    # Mask NaNs or invalids (if needed)
    data = np.ma.masked_invalid(data)

    # Mean burnt area
    mean_burnt_area.append(data.mean())

    # Compute 95th percentile of non-zero values
    nonzero_data = data[data > 0]
    if nonzero_data.count() > 0:
        p95 = np.percentile(nonzero_data.compressed(), 95)
        extreme_data = data[data >= p95]
        mean_extreme_burnt_area.append(extreme_data.mean())
    else:
        mean_extreme_burnt_area.append(0.0)
'''
# (Optional) Plot the time series

fig, ax1 = plt.subplots(figsize=(12, 6))

# First y-axis (left)
ax1.plot(time_points, mean_burnt_area, 'b-')#, label='Mean Burnt Area')
ax1.set_xlabel('Time')
ax1.set_ylabel('Mean Burned Area (%)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Second y-axis (right)
ax2 = ax1.twinx()
ax2.plot(time_points, mean_extreme_burnt_area, 'r-')#, label='Mean Extreme Burned Area (95th percentile)')
ax2.set_ylabel('Extreme Burned Area (%)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Optional: Add legends
fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))

plt.title('Burnt Area Time Series (Mean vs Extreme)')
plt.tight_layout()
plt.show()

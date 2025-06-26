import glob
import os

fact_dir = 'outputs/outputs_scratch/ConFLAME_nrt-attribution9/Amazon-2425/samples/_19-frac_points_0.5/factual-/Evaluate/'
cfact_dir = 'outputs/outputs_scratch/ConFLAME_nrt-attribution9/Amazon-2425/samples/_19-frac_points_0.5/counterfactual-/Evaluate/'

fact_files = sorted(glob.glob(os.path.join(fact_dir, 'sample-pred*.nc')))[0:100]
cfact_files = sorted(glob.glob(os.path.join(cfact_dir, 'sample-pred*.nc')))[0:100]

import iris
import numpy as np

# Load first cube to get grid info
template_cube = iris.load_cube(fact_files[0])
ny, nx = template_cube.shape[-2:]  # latitude, longitude

# Array to store counts
count_map = np.zeros((ny, nx), dtype=int)

# Which month index?
month_idx = 0  # adjust as needed

for f_file, c_file in zip(fact_files, cfact_files):
    fact_cube = iris.load_cube(f_file)
    cfact_cube = iris.load_cube(c_file)
    
    # Extract the month slice
    fact_data = fact_cube.data[month_idx, :, :]
    cfact_data = cfact_cube.data[month_idx, :, :]
    
    # Compare and count
    count_map += (fact_data > cfact_data)

import matplotlib.pyplot as plt

lons = template_cube.coord('longitude').points
lats = template_cube.coord('latitude').points

plt.figure(figsize=(10, 6))
plt.pcolormesh(lons, lats, count_map, shading='auto', cmap='viridis')
plt.colorbar(label='Count of Factual > Counterfactual')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'Count map: factual > counterfactual (month index {month_idx})')
plt.show()

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import os
import seaborn as sns
from pdb import set_trace
# --- Paths ---
dir1 = "outputs/outputs_scratch/ConFLAME_isimip_attribution-pd-large/Amazon-2425/samples/_18-frac_points_0.5/baseline-/control/"
dir2 = "outputs/outputs_scratch/ConFLAME_isimip_attribution-Nopd-bigs/Amazon-2425/samples/_16-frac_points_0.5/baseline-/control/"
pop_file = "data/data/driving_data2425/Amazon/isimp3a/obsclim/GSWP3-W5E5/period_2002_2019/population_density_regridded_to_era5.nc"

# --- List and randomly sample 10 files ---
files1 = sorted(glob.glob(os.path.join(dir1, "sample-pred*.nc")))
files2 = sorted(glob.glob(os.path.join(dir2, "sample-pred*.nc")))

assert len(files1) == len(files2), "Number of paired files mismatch!"

sample_indices = random.sample(range(len(files1)), 10)
sample_files1 = [files1[i] for i in sample_indices]
sample_files2 = [files2[i] for i in sample_indices]

# --- Load population density and take annual mean ---
pop_ds = xr.open_dataset(pop_file)
# Assume variable name is something like "population_density" (adjust if needed)
pop_var = list(pop_ds.data_vars)[0]  # pick first variable
pop = pop_ds[pop_var].mean(dim="time")  # annual mean (2002–2019)

# --- Collect differences ---
x_vals = pop.values.flatten()
y_vals = []

for f1, f2 in zip(sample_files1, sample_files2):
    ds1 = xr.open_dataset(f1)
    ds2 = xr.open_dataset(f2)

    # Assume fire variable is the first variable
    varname = list(ds1.data_vars)[0]
    fire1 = ds1[varname]
    fire2 = ds2[varname]
    
    diff = (fire1 - fire2)#.mean(dim="time")  # mean over time
    y_vals.append(diff.values.flatten())

    ds1.close()
    ds2.close()

# Convert to array
y_vals = np.array(y_vals)  # shape (10, npoints)

# --- Plot ---
plt.figure(figsize=(8,6))

#for i in range(y_vals.shape[0]):
#    plt.scatter(x_vals, y_vals[i], alpha=0.5, label=f"Sample {i+1}")
all_y = y_vals.flatten()
set_trace()
all_x = np.tile(x_vals, y_vals.shape[0])
mask = all_x <= 10000
all_x = all_x[mask]
all_y = all_y[mask]
all_x = np.log(all_x + 1)
plt.figure(figsize=(8,6))
sns.kdeplot(x=all_x, y=all_y, fill=True, cmap="mako", thresh=0.05, levels=100)
#plt.gca().set_xlim([0, 1000])
plt.xlabel("Annual mean population density")
plt.ylabel("Difference in fire output (setup1 - setup2)")
plt.title("KDE density of fire differences vs population density")
plt.show()


'''
plt.xlabel("Annual mean population density")
plt.ylabel("Difference in fire output (setup1 - setup2)")
plt.title("Scatter of fire differences vs population density")
plt.legend()
plt.show()
'''

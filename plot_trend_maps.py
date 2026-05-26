import iris
import numpy as np
import glob

import sys
sys.path.append('SoW_info/')
from state_of_wildfires_colours  import SoW_cmap
from state_of_wildfires_region_info  import get_region_info
sys.path.append('libs/')
from plot_maps import *

from pdb import set_trace

# Path
path = "/data/users/douglas.kelley/fapar_model/outputs/bys/outputs/model_test_baseline_1/samples/_9-frac_points_0.001/baseline-/control/"

files = sorted(glob.glob(path + "sample-pred*.nc"))

diffs = []

for f in files:
    cube = iris.load_cube(f)

    data = cube.data  # shape: (time, lat, lon)

    # Split into halves
    first_half = data[:12, :, :]
    second_half = data[12:, :, :]

    # Mean over time
    first_mean = np.mean(first_half, axis=0)
    second_mean = np.mean(second_half, axis=0)

    # Difference
    diff = second_mean - first_mean

    diffs.append(diff)

# Stack into (n_samples, lat, lon)
diffs = np.stack(diffs, axis=0)

p5  = np.percentile(diffs, 5, axis=0)
p50 = np.percentile(diffs, 50, axis=0)
p95 = np.percentile(diffs, 95, axis=0)

prob_gt0 = np.mean(diffs > 0, axis=0) #* 100  # %
prob_gt0[np.isnan(p50)] = np.nan

template = iris.load_cube(files[0])[0, :, :]  # just lat/lon grid

def make_cube(data, name):
    cube = template.copy(data=data)
    cube.rename(name)
    return cube

p5_cube  = make_cube(p5,  "delta_p5")
p50_cube = make_cube(p50, "delta_p50")
p95_cube = make_cube(p95, "delta_p95")
prob_cube = make_cube(prob_gt0, "prob_delta_gt0")


fig, axes = set_up_sow_plot_windows(2, 1, p50_cube)
cmap = SoW_cmap['diverging_GreenPink']

plot_map_sow(p50_cube, cmap = cmap, ax = axes[0])
plot_map_sow(prob_cube, ax = axes[1], extend = 'neither', levels = [0, 0.01, 0.05, 0.1, 0.9, 0.95, 0.99, 1])


set_trace()

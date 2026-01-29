import iris
import iris.plot as iplt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
sys.path.append('libs/')
from plot_maps import *

import numpy as np
from pdb import set_trace


def fine_max_indicies(cube):
    max_indices = np.argmax(cube.data, axis=0)
    max_cube = cube[0].copy()
    max_cube.data = max_indices
    return max_cube

def return_time_index_value(cube, index_cube, example_cube = None):
         
    if example_cube is not None:
        cube = cube.regrid(example_cube, iris.analysis.Linear())
        time_points = [('time', example_cube.coord('time').points)]
        cube = cube.interpolate(time_points, iris.analysis.Linear())
    else:
        example_cube = cube.copy()
    ntime, nlat, nlon = cube.shape
    lat_idx, lon_idx = np.indices((nlat, nlon))
    
    extracted_data = cube.data[index_cube.data, lat_idx, lon_idx]
               
    cube_out = cube[0].copy()
    cube_out.data = extracted_data
    return cube_out

def find_none_masked_time(cube):
    # 1. Define what "valid data" means for your cube
    def is_valid(data):
        # Returns True for values that are finite (not NaN/Inf) AND not masked
        return np.isfinite(data)

    # 2. Collapse the time dimension using the COUNT aggregator
    # This will return a 2D (lat, lon) cube where each cell is the count of valid times
    valid_counts_cube = cube.collapsed(
        'time', 
        iris.analysis.COUNT, 
        function=is_valid
    )

    # Optional: Rename the cube to reflect its new meaning
    valid_counts_cube.rename('count_of_valid_fire_fraction_times')
    valid_counts_cube.units = '1'
    return valid_counts_cube

def plot_netcdf_files(nc_files, dir, output_path, 
                      whenMax = False, plot_mask = False):
    
    def open_nc(f):
        if f[-3:] != '.nc': f = f + '.nc'
        return iris.load_cube(f)
    cubes = iris.cube.CubeList([open_nc(dir + f) for f in nc_files])
    
    processed_cubes = []
    for cube in cubes:
        if cube.coords("time"):
            if whenMax:
                if 'max_cube' not in locals():
                    max_cube = fine_max_indicies(cube)
                    cube0 = cube.copy()
                cube_out = return_time_index_value(cube, max_cube, cube0)
            elif plot_mask:
                cube_out = find_none_masked_time(cube)
            else:
                cube_out = cube.collapsed("time", iris.analysis.MEAN)
        processed_cubes.append(cube_out)
    plot_maps(processed_cubes, nc_files, output_path)

def plot_maps(cubes, titles, output_path):
    # Step 3: Set up subplots with Cartopy
    # Determine number of rows and columns
    
    n_plots = len(cubes)
    n_rows  = int(np.ceil(np.sqrt(n_plots)))
    n_cols  = int(np.ceil(n_plots/n_rows))
    
    fig, axes = set_up_sow_plot_windows(n_rows, n_cols, cubes[0], 
                                        size_scale = 3)
    
    for cube, ttl, ax in zip(cubes, titles, axes):
        try:
            plot_map_sow(cube, ttl, cmap=SoW_cmap['gradient_hues'], ax = ax)
        except:
            if cube.shape[0] == 1:
                plot_map_sow(cube[0], ttl, cmap=SoW_cmap['gradient_hues'], 
                             ax = ax)
            else:
                set_trace()
    
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    
if __name__=="__main__":
    # Step 1: Load all NetCDF files
    
    dir = 'data/data/driving_data/Global/isimp3a/obsclim/GSWP3-W5E5/period_2010_2012/masked/'
    
    nc_files = ["pr_mean.nc", "tas_max.nc", "Tree_cover_vcf.nc", 
                "Total_cover_vcf.nc", "vpd_mean.nc", "cveg.nc", "burnt_area.nc"]  

    plot_netcdf_files(nc_files, dir)



def plot_ensemble_maps(cubes, titles = None, 
                       control_colours = None, 
                       output_path = 'figs/unnamed-ensemble.png',
                       percentiles = [5, 95], *args, **kw):
    
    def get_percentiles(cube):
        if isinstance(cube, list) or isinstance(cube, tuple):
            cube = cube[0]
        out = cube.collapsed('time', iris.analysis.PERCENTILE, 
                             percent = percentiles)
        out = out.collapsed('realization', 
                             iris.analysis.PERCENTILE, 
                             percent = percentiles)
        return out
    
    cube_pc = [get_percentiles(cube) for cube in cubes]
    
    n_plots = len(cubes)
    
    n_rows = int(np.ceil(np.sqrt(n_plots)))
    n_cols = len(percentiles) * int(np.ceil(n_plots/n_rows))
    n_rows *= len(percentiles)

    fig, axes = set_up_sow_plot_windows(n_rows, n_cols, cubes[0][0][0], size_scale = 3)
    nplt = 0
    if control_colours is None:
       control_colours = ['gradient_hues'] * len(cubes)

    for cube, cmap, ttl in zip(cube_pc, control_colours, titles):
        for i in range(len(percentiles)):
            for j in range(len(percentiles)):
                title = ttl + ' ' + str(percentiles[i]) + \
                        '%ile\nof the ' + \
                        str(percentiles[j]) + '%ile over time'
                
                plot_map_sow(cube[i][j], title, cmap=SoW_cmap[cmap],  ax = axes[nplt])
                nplt += 1

    fig.savefig(output_path, dpi=300, bbox_inches="tight")


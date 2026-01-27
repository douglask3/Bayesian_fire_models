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


def plot_netcdf_files(nc_files, dir, output_path, whenMax = False):
    
    def open_nc(f):
        if f[-3:] != '.nc': f = f + '.nc'
        return iris.load_cube(f)
    cubes = iris.cube.CubeList([open_nc(dir + f) for f in nc_files])
    
    processed_cubes = []
    for cube in cubes:
        if cube.coords("time"):
            if whenMax:
                if 'max_cube' not in locals():
                    max_indices = np.argmax(cube.data, axis=0)
                    max_cube = cube[0].copy()
                    max_cube.data = max_indices
                    cube0 = cube.copy()
        
                cube = cube.regrid(cube0, iris.analysis.Linear())
                time_points = [('time', cube0.coord('time').points)]
                cube = cube.interpolate(time_points, iris.analysis.Linear())
                ntime, nlat, nlon = cube.shape
                lat_idx, lon_idx = np.indices((nlat, nlon))
                try:
                    extracted_data = cube.data[max_indices, lat_idx, lon_idx]
                except:
                    set_trace()
                cube_out = cube[0].copy()
                cube_out.data = extracted_data
            else:
                cube_out = cube.collapsed("time", iris.analysis.MEAN)
        processed_cubes.append(cube_out)

    # Step 3: Set up subplots with Cartopy
    # Determine number of rows and columns
    n_plots = len(processed_cubes)
    n_rows = int(np.ceil(np.sqrt(n_plots)))
    n_cols = int(np.ceil(n_plots/n_rows))
    
    fig, axes = set_up_sow_plot_windows(n_rows, n_cols, processed_cubes[0], size_scale = 3)
    
    for cube, ttl, ax in zip(processed_cubes, nc_files, axes):
        
        try:
            plot_map_sow(cube, ttl, cmap=SoW_cmap['gradient_hues'], ax = ax)
        except:
            if cube.shape[0] == 1:
                plot_map_sow(cube[0], ttl, cmap=SoW_cmap['gradient_hues'], ax = ax)
            else:
                exit()
    
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return()
    #n_rows, n_cols = 3, 3  # Adjust based on the number of plots needed
    

    #fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 7), 
    #                         subplot_kw={'projection': ccrs.PlateCarree()}, 
    #                         constrained_layout=True)

    # Flatten the axes array for easy iteration
    #axes = axes.flatten()
    
    # Loop through the data and plot
    for i, (ax, cube, title) in enumerate(zip(axes, processed_cubes, nc_files)):
        ax.set_title(title)
        ax.coastlines()
        
        # Plot the cube
        im = iplt.pcolormesh(cube, axes=ax)
        
        # Add a colorbar
        fig.colorbar(im, ax=ax, orientation="vertical")
    
    # Hide any unused subplots if fewer than 9 plots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    set_trace()
    fig.savefig("outputs/outputs/ar7_annual_averages.png", dpi=300, bbox_inches="tight")


if __name__=="__main__":
    # Step 1: Load all NetCDF files
    
    dir = 'data/data/driving_data/Global/isimp3a/obsclim/GSWP3-W5E5/period_2010_2012/masked/'
    
    nc_files = ["pr_mean.nc", "tas_max.nc", "Tree_cover_vcf.nc", 
                "Total_cover_vcf.nc", "vpd_mean.nc", "cveg.nc", "burnt_area.nc"]  

    plot_netcdf_files(nc_files, dir)

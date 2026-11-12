import iris
import iris.plot as iplt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pdb import set_trace


def plot_netcdf_files(nc_files, dir):
    set_trace() 
    cubes = iris.cube.CubeList([iris.load_cube(dir + f) for f in nc_files])
    processed_cubes = []
    for cube in cubes:
        if cube.coords("time"):
            cube = cube.collapsed("time", iris.analysis.MEAN)
        processed_cubes.append(cube)

    # Step 3: Set up subplots with Cartopy
    # Determine number of rows and columns
    n_rows, n_cols = 3, 3  # Adjust based on the number of plots needed
    n_plots = len(processed_cubes)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 7), 
                             subplot_kw={'projection': ccrs.PlateCarree()}, 
                             constrained_layout=True)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
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

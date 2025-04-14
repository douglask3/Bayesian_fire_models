import iris
import iris.coord_categorisation as icat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import geopandas as gpd
import calendar
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon
from pdb import set_trace


def open_netcdf_and_find_clim(filename):
    # Load NetCDF file
    cube = iris.load_cube(filename)
    
    # Ensure time coordinate is properly formatted
    time_coord = cube.coord("time") 
    time_units = time_coord.units

    dates = time_units.num2date(time_coord.points)
    last_date = dates[-1]
    start_date = last_date.replace(year=last_date.year - 1)
    
    
    # Add month categorisation to time
    icat.add_month(cube, "time", name="month")
    
    # Extract last 12 months of data    
    constraint = iris.Constraint(
        time=lambda cell: start_date <= cell.point <= last_date
    )
    last_12_months_cube = cube.extract(constraint)[1:]
    
    # Compute the climatological mean for each month
    climatology = cube.aggregated_by("month", iris.analysis.MEAN)
    
    # Compute anomaly for the most recent year
    last_year_constraint = iris.Constraint(
        time=lambda cell: last_date.replace(year=last_date.year - 1) <= cell.point <= last_date
    )
    
    last_year_cube = cube.extract(last_year_constraint)[1:]
    
    anomaly = last_year_cube.copy()
    anomaly.data = anomaly.data - climatology.data
    return anomaly, climatology

def load_shapefile(shapefile_path):
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)  # Load shapefile
    gdf["geometry"] = gdf["geometry"].buffer(0)
    shapefile_geometries = unary_union(gdf.geometry)  # Merge all geometries    
    if isinstance(shapefile_geometries, MultiPolygon):
        shapefile_geometries = list(shapefile_geometries.geoms)  # Extract individual polygons
    return shapefile_geometries

def plot_all_climatology(climatology, title="Annual Mean Burnt Area per Month",
                         fig_id="burnt_area_climatology", cmap='Oranges', 
                         c_bins = [0, 0.1, 0.5, 1, 2, 5, 10], extend = 'max'):
    fig, axes = plt.subplots(3, 4, figsize=(12, 9),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    # Compute global min/max for consistent colour scale
    clim_min, clim_max = np.min(climatology.data), np.max(climatology.data)
    
    
    norm = mcolors.BoundaryNorm(boundaries=c_bins, ncolors=len(c_bins)-1, clip = False)
    cmap = plt.get_cmap(cmap, len(c_bins)-1)

    # Load shapefile
    shp = shpreader.Reader("data/data/SoW2425_shapes/SoW2425_Focal_MASTER_20250221.shp")
    shapefile_geometries = list(shp.geometries())

    for i, month_cube in enumerate(climatology.slices_over("month")):
        ax = axes.flat[i]
        month_name = month_cube.coord("month").points[0]  # Real month index

        im = ax.pcolormesh(month_cube.coord("longitude").points, 
                           month_cube.coord("latitude").points, 
                           month_cube.data, 
                           transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)

        # Geographic features
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS, edgecolor='blue', alpha=0.33)

        # Shapefile outline
        ax.add_geometries(shapefile_geometries, ccrs.PlateCarree(), edgecolor='black', 
                          facecolor='none', linewidth=1)

        ax.set_title(month_name)  # Use real month name

    # Single colourbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal',
                        ticks=c_bins, fraction=0.05, pad=0.1, extend = extend)
    cbar.set_label("Burnt Area")

    plt.suptitle(Region_title + " " + title)
    
    plt.savefig(f"figs/{Region_title}-{fig_id}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"figs/{Region_title}-{fig_id}.pdf", dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

def plot_region(filename, shapefile_path, cmap, dcmap, levels, dlevels):
    anomaly, climatology = open_netcdf_and_find_clim(filename)
    shapefile_geometries = load_shapefile(shapefile_path)
    
    custom_cmap = LinearSegmentedColormap.from_list("cmap_interp", cmap, N=len(levels))
    plot_all_climatology(climatology, cmap = custom_cmap, c_bins = levels,
                        extend='max')

    custom_cmap = LinearSegmentedColormap.from_list("dcmap_interp", dcmap[1:-1], 
                                                    N = len(dlevels)-1)  
    custom_cmap.set_under(dcmap[0]) 
    custom_cmap.set_over(dcmap[-1])
    plot_all_climatology(anomaly, "Burnt Area Anomaly Mar 24 - Feb 25", "burnt_area_anaomoly",
                     cmap = custom_cmap, c_bins = dlevels,
                     extend='both')

# Discrete colormap with same number of levels
SoW_gradient_red = [
    "#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84",
    "#fc8d59", "#ef6548", "#d7301f", "#b30000", "#7f0000"
]

SoW_diverging_TealOrange = ["#004c4b", "#008786", "#6bbbaf", "#b6e0db", "#ffffff", "#ffd8b8", "#ffb271", "#e57100", "#8a3b00"]



if __name__=="__main__":    
    shapefile_path = "data/data/SoW2425_shapes/SoW2425_Focal_MASTER_20250221.shp"

    Region_title = "Pantanal" 
    filename = "data/data/driving_data2425/Pantanal/burnt_area.nc"

    cmap = SoW_gradient_red
    dcmap = SoW_diverging_TealOrange
    levels = [0, 0.1, 0.5, 1, 2, 5, 10]
    dlevels = [-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10]
    plot_region(filename, shapefile_path, cmap, dcmap, levels, dlevels)

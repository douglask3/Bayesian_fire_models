
filename = "data/data/driving_data2425/Pantanal/burnt_area.nc"
import iris
import iris.coord_categorisation as icat
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import geopandas as gpd
import calendar
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon
from pdb import set_trace


Region_title = "Pantanal" 

# Load NetCDF file
cube = iris.load_cube(filename)

# Ensure time coordinate is properly formatted
time_coord = cube.coord("time")
time_units = time_coord.units

dates = time_units.num2date(time_coord.points)
last_date = dates[-1]
start_date = last_date.replace(year=last_date.year - 1)

# Extract last 12 months of data

constraint = iris.Constraint(
    time=lambda cell: start_date <= cell.point <= last_date
)

last_12_months_cube = cube.extract(constraint)[1:]
set_trace()
# Add month categorisation to time
icat.add_month(last_12_months_cube, "time", name="month")

# Compute the climatological mean for each month
climatology = last_12_months_cube.aggregated_by("month", iris.analysis.MEAN)

# Compute anomaly for the most recent year
last_year_constraint = iris.Constraint(
    time=lambda cell: last_date.replace(year=last_date.year - 1) <= cell.point <= last_date
)

last_year_cube = cube.extract(last_year_constraint)[1:]
anomaly = last_year_cube.collapsed("time", iris.analysis.MEAN) - climatology

# Load the shapefile
shapefile_path = "data/data/SoW2425_shapes/SoW2425_Focal_MASTER_20250221.shp"
gdf = gpd.read_file(shapefile_path)  # Load shapefile
gdf["geometry"] = gdf["geometry"].buffer(0)
shapefile_geometries = unary_union(gdf.geometry)  # Merge all geometries
if isinstance(shapefile_geometries, MultiPolygon):
    shapefile_geometries = list(shapefile_geometries.geoms)  # Extract individual polygons


def plot_all_climatology(climatology, title="Annual Mean Burnt Area per Month",
                         fig_id="burnt_area_climatology", cmap='Oranges', symmetric=False):
    fig, axes = plt.subplots(3, 4, figsize=(12, 9),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    # Compute global min/max for consistent colour scale
    clim_min, clim_max = np.min(climatology.data), np.max(climatology.data)
    
    # Set norm (nonlinear for better visualization)
    if symmetric:
        norm = mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=-clim_max, vmax=clim_max, base=10)
    else:
        norm = mcolors.LogNorm(vmin=max(clim_min, 1e-2), vmax=clim_max)  # Avoid log(0)

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
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label("Burnt Area")

    plt.suptitle(Region_title + " " + title)
    #plt.tight_layout()
    plt.savefig(f"figs/{Region_title}-{fig_id}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"figs/{Region_title}-{fig_id}.pdf", dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()


plot_all_climatology(climatology)
plot_all_climatology(anomaly, "Burnt Area Anomaly in Last Year", "burnt_area_anaomoly",
                     cmap='RdBu_r', symmetric = True)
set_trace()

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})

im = ax.pcolormesh(anomaly.coord("longitude").points, 
                    anomaly.coord("latitude").points, 
                    anomaly.data, 
                    transform=ccrs.PlateCarree(), cmap='RdBu_r')

# Add geographic features
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=0.5)  # Country borders
ax.add_feature(cfeature.LAKES, alpha=0.5)  # Lakes
ax.add_feature(cfeature.RIVERS, linestyle='-', edgecolor='blue')  # Major rivers

ax.set_title(Region_title + " Burnt Area Anomaly in Last Year")
plt.colorbar(im, ax=ax, orientation='horizontal')
plt.savefig("figs/" + Region_title + "-burnt_area_anaomoly.png", 
            dpi=300, bbox_inches='tight')  # Save as PNG
plt.savefig("figs/" + Region_title + "-burnt_area_anaomoly.pdf", dpi=300, bbox_inches='tight')  # Save as PDF
plt.clf()  
plt.close()

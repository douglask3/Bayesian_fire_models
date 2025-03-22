
filename = "data/data/driving_data2425/Pantanal/burnt_area.nc"
import iris
import iris.coord_categorisation as icat
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import calendar
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


fig, axes = plt.subplots(3, 4, figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})

for i, month_cube in enumerate(climatology.slices_over("month")):
    ax = axes.flat[i]

    # Extracting month index (1-12) and getting the real month name
    #set_trace()
    #month_index = int(month_cube.coord("month").points[0])
    month_name = month_cube.coord("month").points[0]#calendar.month_name[month_index]  # Converts 1->"January", 2->"February", etc.

    im = ax.pcolormesh(month_cube.coord("longitude").points, 
                        month_cube.coord("latitude").points, 
                        month_cube.data, 
                        transform=ccrs.PlateCarree(), cmap='Oranges')

    # Add geographic features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=0.5)  # Country borders
    ax.add_feature(cfeature.LAKES, alpha=0.5)  # Lakes
    ax.add_feature(cfeature.RIVERS, linestyle='-', edgecolor='blue', alpha=0.33)  # Major rivers

    # Add shapefile outline
    ax.add_geometries(shapefile_geometries, ccrs.PlateCarree(), edgecolor='black', 
                      facecolor='none', linewidth=1)
    

    ax.set_title(month_name)  # Use real month name
    plt.colorbar(im, ax=ax, orientation='horizontal')

plt.suptitle(Region_title + " Anual Mean Burnt Area per Month")
plt.tight_layout()
plt.savefig("figs/" + Region_title + "-burnt_area_climatology.png", 
            dpi=300, bbox_inches='tight')  # Save as PNG
plt.savefig("figs/" + Region_title + "-burnt_area_climatology.pdf", dpi=300, bbox_inches='tight')  # Save as PDF
plt.clf()  
plt.close()
# Plot anomaly
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


filename = "data/data/driving_data2425/Pantanal/burnt_area.nc"
import iris
import iris.coord_categorisation as icat
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pdb import set_trace
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

# Plot maps
fig, axes = plt.subplots(3, 4, figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})
for i, month_cube in enumerate(climatology.slices_over("month")):
    ax = axes.flat[i]
    im = ax.pcolormesh(month_cube.coord("longitude").points, 
                        month_cube.coord("latitude").points, 
                        month_cube.data, 
                        transform=ccrs.PlateCarree(), cmap='Oranges')
    ax.coastlines()
    ax.set_title(f"Month {i+1}")
    plt.colorbar(im, ax=ax, orientation='horizontal')

plt.suptitle("Annual Mean Burnt Area per Month")
plt.tight_layout()
plt.show()

# Plot anomaly
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
im = ax.pcolormesh(anomaly.coord("longitude").points, 
                    anomaly.coord("latitude").points, 
                    anomaly.data, 
                    transform=ccrs.PlateCarree(), cmap='RdBu_r')
ax.coastlines()
ax.set_title("Burnt Area Anomaly in Last Year")
plt.colorbar(im, ax=ax, orientation='horizontal')
plt.show()


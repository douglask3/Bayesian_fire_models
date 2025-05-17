import iris
import numpy as np
import cftime

target_dir = "data/data/driving_data2425/Amazon/nrt/era5_monthly/"
# Load cubes
hyde_cube = iris.load_cube('data/data/HYDE/cropland.nc')
era5_cube = iris.load_cube(target_dir + 'precip.nc')


# Function to convert a cftime object to a decimal year
def to_decimal_year(dt):
    year = dt.year
    start_of_year = cftime.datetime(year, 1, 1, calendar=dt.calendar)
    end_of_year = cftime.datetime(year + 1, 1, 1, calendar=dt.calendar)
    year_length = (end_of_year - start_of_year).days
    decimal = year + (dt - start_of_year).days / year_length
    return decimal

# Extract time coordinates
hyde_time_coord = hyde_cube.coord('time')
era5_time_coord = era5_cube.coord('time')

# Convert all time points to date objects using their own units
hyde_dates = hyde_time_coord.units.num2date(hyde_time_coord.points)
era5_dates = era5_time_coord.units.num2date(era5_time_coord.points)

# Convert to decimal years
hyde_years = np.array([to_decimal_year(dt) for dt in hyde_dates])
era5_years = np.array([to_decimal_year(dt) for dt in era5_dates])

# Get ERA5 time range in decimal years
start_year = era5_years.min()
end_year = era5_years.max()

# Mask HYDE cube to this range
mask = (hyde_years >= start_year) & (hyde_years <= end_year)
hyde_cube_subset = hyde_cube[mask, :, :]


# Step 2: Crop HYDE spatially to the ERA5 extent
lat_constraint = iris.Constraint(latitude=lambda lat: era5_cube.coord('latitude').points.min() <= lat <= era5_cube.coord('latitude').points.max())
lon_constraint = iris.Constraint(longitude=lambda lon: era5_cube.coord('longitude').points.min() <= lon <= era5_cube.coord('longitude').points.max())
hyde_cube_cropped = hyde_cube_subset.extract(lat_constraint & lon_constraint)

# Step 3: Regrid HYDE to ERA5 grid
# First make sure both have compatible coordinate systems
hyde_cube_cropped.coord('latitude').guess_bounds()
hyde_cube_cropped.coord('longitude').guess_bounds()
era5_cube.coord('latitude').guess_bounds()
era5_cube.coord('longitude').guess_bounds()

# Regrid HYDE to match ERA5 using bilinear interpolation
hyde_regridded = hyde_cube_cropped.regrid(era5_cube, iris.analysis.Linear())

# Optional: Save output
iris.save(hyde_regridded, target_dir + "cropland_regridded_to_era5.nc")


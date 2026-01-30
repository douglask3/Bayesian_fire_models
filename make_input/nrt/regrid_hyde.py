import iris
import numpy as np
import cftime
from iris.analysis import Nearest, Linear
from pdb import set_trace

import iris
import numpy as np
from scipy.interpolate import griddata


# Function to convert a cftime object to a decimal year
def to_decimal_year(dt):
    year = dt.year
    start_of_year = cftime.datetime(year, 1, 1, calendar=dt.calendar)
    end_of_year = cftime.datetime(year + 1, 1, 1, calendar=dt.calendar)
    year_length = (end_of_year - start_of_year).days
    decimal = year + (dt - start_of_year).days / year_length
    return decimal


def extrapolate_masked(cube, method='nearest'):
    """
    Extrapolate masked values in an Iris cube spatially.

    Parameters
    ----------
    cube : iris.cube.Cube
        Cube with masked values to extrapolate.
    method : str, optional
        Interpolation method. Options: 'nearest', 'linear', 'cubic'.
        Default is 'nearest'.

    Returns
    -------
    new_cube : iris.cube.Cube
        Cube with masked values filled.
    """
    # Get data and mask
    data = cube.data.copy()
    mask = np.ma.getmaskarray(data)
    
    if not np.any(mask):
        print("No masked values found.")
        return cube.copy()

    # Get latitude and longitude
    lats = cube.coord('latitude').points
    lons = cube.coord('longitude').points
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Only use unmasked points for interpolation
    points = np.column_stack((lat_grid[~mask], lon_grid[~mask]))
    values = data[~mask]

    # Interpolate over masked points
    filled_values = griddata(points, values,
                             (lat_grid[mask], lon_grid[mask]),
                             method=method)

    # Fill the masked cells
    data_filled = data.copy()
    data_filled[mask] = filled_values

    # Create a new cube with filled data
    new_cube = cube.copy(data=data_filled)
    return new_cube

def extrapolate_masked_timeseries(cube, method='nearest'):
    """
    Extrapolate masked values in a 3D Iris cube with time, latitude, longitude.
    
    Parameters
    ----------
    cube : iris.cube.Cube
        Cube with shape (time, lat, lon) and masked values.
    method : str, optional
        Interpolation method. Options: 'nearest', 'linear', 'cubic'.
        Default is 'nearest'.
    
    Returns
    -------
    new_cube : iris.cube.Cube
        Cube with masked values filled in all time slices.
    """
    data_filled = cube.data.copy()
    
    for t in range(cube.shape[0]):
        # Extract 2D slice
        slice_2d = cube[t, :, :]
        filled_slice = extrapolate_masked(slice_2d, method=method)
        data_filled[t, :, :] = filled_slice.data
    
    # Return new cube
    new_cube = cube.copy(data=data_filled)
    return new_cube


# Load cubes
def regrid_hyde_cube(hyde_file, target_file, hyde_dir = "", target_dir = ""):
    hyde_cube = iris.load_cube(hyde_dir + hyde_file)
    era5_cube = iris.load_cube(target_dir + target_file)
    
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
    hyde_regridded = extrapolate_masked_timeseries(hyde_regridded)
    #set_trace()
    #hyde_filled = hyde_regridded.regrid(hyde_regridded, Linear(extrapolation_mode='extrapolate'))
    
    # Optional: Save output
    iris.save(hyde_regridded, target_dir + hyde_file[:-3] + "_regridded_to_era5.nc")
hyde_dir = "data/data/HYDE/"
hyde_files = ["cropland.nc", "grazing_land.nc", "pasture.nc", "population_density.nc", 
                  "rangeland.nc", "rural_population.nc", "total_irrigated.nc", "urban_area.nc",
                  "urban_population.nc"]
target_file = 'precip.nc'

def regrid_hyde_for_regions(regions, hyde_files, target_file, hyde_dir, target_dir):
    for region in regions:
        target_dir_r = target_dir + region + "/nrt/era5_monthly/"
        for hyde_file in hyde_files:
            regrid_hyde_cube(hyde_file, target_file, hyde_dir, target_dir_r)

if __name__=="__main__":
    regions = ["Amazon", "Congo", "LA", "Pantanal", "Alberta", "NEIndia"]
    target_dir = "data/data/driving_data2425/"
    regrid_hyde_for_regions(regions, hyde_files, target_file, hyde_dir, target_dir)
    

    


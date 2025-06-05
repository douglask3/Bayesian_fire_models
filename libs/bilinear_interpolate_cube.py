
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import iris
import iris.coords as icoords
import iris.cube as icube
from scipy.ndimage import gaussian_filter


def smooth_cube(cube, sigma=1):
    smoothed = cube.copy()
    smoothed.data = gaussian_filter(smoothed.data, sigma=sigma)
    return smoothed

def bilinear_interpolate_cube(cube, factor=10):
    """
    Upscales a 2D Iris cube using bilinear interpolation on the lat/lon grid.

    Parameters:
    -----------
    cube : iris.cube.Cube
        An Iris cube with dimensions (latitude, longitude).
    factor : int
        Factor by which to increase resolution along each spatial dimension.

    Returns:
    --------
    iris.cube.Cube
        A new cube with bilinearly interpolated data on a higher-resolution grid.
    """
    # Extract data and coordinates
    data = cube.data
    lat = cube.coord('latitude').points
    lon = cube.coord('longitude').points

    # Check dimensionality
    if data.ndim != 2 or cube.coord_dims('latitude') != (0,) or cube.coord_dims('longitude') != (1,):
        raise ValueError("This function only supports 2D cubes with (latitude, longitude) dimensions in that order.")

    # Create new high-res coordinate arrays
    new_lat = np.linspace(lat.min(), lat.max(), len(lat) * factor)
    new_lon = np.linspace(lon.min(), lon.max(), len(lon) * factor)
    new_lon_grid, new_lat_grid = np.meshgrid(new_lon, new_lat)

    # Prepare the interpolator
    interpolator = RegularGridInterpolator((lat, lon), data, method='linear', bounds_error=False, fill_value=np.nan)

    # Interpolate
    interp_points = np.array([new_lat_grid.flatten(), new_lon_grid.flatten()]).T
    new_data = interpolator(interp_points).reshape((len(new_lat), len(new_lon)))

    # Create new dimension coordinates
    lat_coord = icoords.DimCoord(new_lat, standard_name='latitude', units=cube.coord('latitude').units)
    lon_coord = icoords.DimCoord(new_lon, standard_name='longitude', units=cube.coord('longitude').units)

    # Create the new cube
    new_cube = icube.Cube(
        new_data,
        long_name=cube.long_name,
        standard_name=cube.standard_name,
        var_name=cube.var_name,
        units=cube.units,
        attributes=cube.attributes,
        dim_coords_and_dims=[(lat_coord, 0), (lon_coord, 1)]
    )

    return new_cube


import iris
import iris.analysis
import iris.coords

import cftime
import cf_units
import numpy as np
import glob
import iris.analysis.cartography as cart

import sys
sys.path.append('libs/')
sys.path.append('make_inputs/nrt/')
from make_ConFLAME_inputs import *
from pdb import set_trace

filename = "data/data/burnt_area/BA_Total/BA_Qdeg_C6_*.hdf5"

def make_layer(file):
    print(file)
    cube = iris.load_cube(file)
    
    nlat = cube.shape[0]
    nlon = cube.shape[1]
    cube.data = np.flip(cube.data, axis=0)
    latitudes = np.linspace(-90 + 180/(2*nlat), 90 - 180/(2*nlat), nlat)
    longitudes = np.linspace(-180 + 360/(2*nlon), 180 - 360/(2*nlon), nlon)

    lat_coord = iris.coords.DimCoord(
        latitudes,
        standard_name='latitude',
        units='degrees'
    )

    lon_coord = iris.coords.DimCoord(
        longitudes,
        standard_name='longitude',
        units='degrees'
    )
    cube.add_dim_coord(lat_coord, 0)  # latitude axis
    cube.add_dim_coord(lon_coord, 1)  # longitude axis
    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()

    year = int(file[-11:-7])
    month = int(file[-7:-5])
    
    time_unit = cf_units.Unit('days since 1970-01-01 00:00:00', calendar='standard')
    cube = add_time_based_on_mnth_year([cube], [year], [month], time_unit)
    cube.data *= 0.213444
    
    grid_areas = cart.area_weights(cube)
    grid_areas_km2 = grid_areas / 1e6
    cube = cube / grid_areas_km2
    cube.units = '1'  # dimensionless
    
    return cube

files = sorted(glob.glob(filename))

cubes = [make_layer(file) for file in files]
cubes = iris.cube.CubeList(cubes).concatenate_cube()
iris.save(cubes, "data/data/burned_area_global.nc")


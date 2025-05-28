import sys
sys.path.append('../libs/')
sys.path.append('libs/')

from constrain_cubes_standard import *

import iris
import iris.coord_systems as cs
import iris.fileformats.netcdf as netcdf

import numpy as np

import iris.quickplot as qplt

file_path = "/data/users/dkelley/MaxEnt/raw_data/grip4_total_dens_m_km2.asc"
output_path = "../../ConFIRE_attribute/isimip3a/driving_data/GSWP3-W5E5-20yrs/Brazil/AllConFire_2000_2009/"
target_cube = iris.load_cube('../data/BR_Biomes.nc')

coord_sys = cs.GeogCS(iris.fileformats.pp.EARTH_RADIUS)

data = np.loadtxt(file_path, skiprows=6)  # Skip the header rows

header_info = {}
with open(file_path, 'r') as file:
    for _ in range(6):
        line = file.readline().strip().split()
        header_info[line[0].lower()] = line[1]

def grabVar(var): return float(header_info[var]) 


dcell = grabVar('cellsize')
lat0 = grabVar('yllcorner') + dcell/2
lat1 = lat0 + grabVar('nrows') * dcell

lon0 = grabVar('xllcorner') + dcell/2
lon1 = lon0 + grabVar('ncols') * dcell

latitude = iris.coords.DimCoord(np.flip(np.arange(lat0, lat1, dcell)), 
                                standard_name='latitude', units='degrees')
longitude = iris.coords.DimCoord(np.arange(lon0, lon1, dcell), 
                                 standard_name='longitude', units='degrees')



# Create a new Iris cube

cube = iris.cube.Cube(data, dim_coords_and_dims=[(latitude, 0), (longitude, 1)],
                      long_name='Road Density', units='km^2')
cube.data[cube.data < -10.0] = np.nan
cube.data[cube.data <   0.0] = 0.0

# Regrid the original cube to the target cube's grid
regridded_cube = cube.regrid(target_cube, iris.analysis.Linear())

iris.save(regridded_cube, output_path + '/' + 'road_density.nc')



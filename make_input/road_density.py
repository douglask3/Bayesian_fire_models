import sys
sys.path.append('../libs/')
sys.path.append('libs/')

from constrain_cubes_standard import *

import iris
import iris.coord_systems as cs
import iris.fileformats.netcdf as netcdf

import numpy as np

import iris.quickplot as qplt

file_path = "data/data/roadDensity/grip4_total_dens_m_km2.asc"
#output_path = "../../ConFIRE_attribute/isimip3a/driving_data/GSWP3-W5E5-20yrs/Brazil/AllConFire_2000_2009/"
#

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
cube_rd = iris.cube.Cube(data, dim_coords_and_dims=[(latitude, 0), (longitude, 1)],
                      long_name='Road Density', units='km^2')
cube_rd.data[cube_rd.data < -10.0] = np.nan
cube_rd.data[cube_rd.data <   0.0] = 0.0

def make_road_density_for_region(region, target_dir, subdir, target_file):
    dir = target_dir + '/' + region + '/' + subdir + '/'
    file = dir + target_file
    #set_trace()
    target_cube = iris.load_cube(file)
    # Regrid the original cube to the target cube's grid
    regridded_cube = cube_rd.regrid(target_cube, iris.analysis.Linear())

    iris.save(regridded_cube, dir + '/' + 'road_density.nc')
    
if __name__=="__main__":
    regions = ['Pantanal', 'Amazon']
    target_dir = "data/data/driving_data_base/"
    subdir = "/nrt/era5_monthly/"
    target_file = "precip.nc"
    for region in regions:
        make_road_density_for_region(region, target_dir, subdir, target_file)
    


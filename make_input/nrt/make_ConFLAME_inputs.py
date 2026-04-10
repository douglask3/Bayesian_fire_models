import os.path
import os
from pathlib import Path

import numpy as np
import iris

import iris.coord_categorisation as icc

import iris.quickplot as qplt
import matplotlib.pyplot as plt
from pdb import set_trace

import glob
import sys
sys.path.append('libs/')
sys.path.append('make_inputs/nrt/')
from constrain_cubes_standard import *
from make_factual_counter import *
from fix_joeys_weird_time_coords import *

import cf_units
import datetime
from iris.coords import DimCoord

def regrid_to_eg_cube(cube, eg_cube):
    if cube.ndim == 3:
        if cube.shape[1:] == eg_cube.shape: 
            return cube
        lat_name = cube.coords()[1].name()
        lon_name = cube.coords()[2].name()
    elif cube_ndim == 2:
        if cube.shape == eg_cube.shape: 
            return cube
        lat_name = cube.coords()[0].name()
        lon_name = cube.coords()[1].name()
    else:
        set_trace()
    cube.coord(lat_name).rename(eg_cube.coords()[0].name())
    cube.coord(lon_name).rename(eg_cube.coords()[1].name())

    return cube.regrid(eg_cube, iris.analysis.Linear())
    
def combine_2d_cubes(cubes, files, time_unit):
    years = [int(file[-10:-6]) for file in files]
    months = [int(file[-5:-3]) for file in files]

    unit = cf_units.Unit(time_unit, calendar='gregorian')
    
    datetimes = [
        datetime.datetime(y, m, 15)
        for y, m in zip(years, months)
    ]
    time_points = unit.date2num(datetimes)

    new_cubes = []
    for cube, t in zip(cubes, time_points):
        
       # Fix latitude
        if cube.coords('lat'):
            coord = cube.coord('lat')
            coord.rename('latitude')
            coord.standard_name = 'latitude'
            coord.var_name = 'lat'
            coord.unit = 'degrees'
    
        # Fix longitude
        if cube.coords('lon'):
            coord = cube.coord('lon')
            coord.rename('longitude')
            coord.standard_name = 'longitude'
            coord.var_name = 'lon'
            coord.unit = 'degrees'

        time_coord = DimCoord(
            [t],
            standard_name='time',
            units=unit
        )
        cube.add_aux_coord(time_coord)
        cube = iris.util.new_axis(cube, 'time')  # promote to dimension
        new_cubes.append(cube)
    iris.util.equalise_attributes(new_cubes)
    
    return iris.cube.CubeList(new_cubes).concatenate_cube()

def combine_3d_cubes(cubes, region, files):
    vname = cubes[0].name()
    cubes0 = cubes.copy()
    try:
        target_time_unit = cubes[0].coord('time').units
    except:
        set_trace()
    new_cubes = []
    for cube, file in zip(cubes, files):
        print(file) 
        
        cube = contrain_to_sow_shapefile(cube, "data/data/driving_data2526/Focal_regions/SoW2526_Focal_MASTER_20260218.shp", region.replace('_', ' '))
        
        cube.data = cube.data.astype(np.float32)
        cube.rename(vname)
        
        try:
            cube.coord('time').convert_units(target_time_unit)
        except:
            cube = fix_joeys_weird_time_coords(cube, file)
        
        cube.coord('time').convert_units(target_time_unit)
        if file == files[0]:
            template_coord = cube.coord('time')
        else:
            target_coord = cube.coord('time')
            target_coord.units = template_coord.units
            target_coord.standard_name = template_coord.standard_name
            target_coord.long_name = template_coord.long_name
            target_coord.var_name = template_coord.var_name
            target_coord.attributes = template_coord.attributes.copy()
            target_coord.coord_system = template_coord.coord_system
        cube.coord('time').points = cube.coord('time').points.astype('float64')
        
        new_cubes.append(cube)
            
    cubes = iris.cube.CubeList(new_cubes)
    iris.util.equalise_attributes(cubes)
    
    # 1. Collect all time coordinates from all cubes
    time_coords = [cube.coord('time') for cube in cubes]
    iris.util.equalise_attributes(time_coords)
    try:
        final_cube = cubes.concatenate_cube()
    except:
        try:
            new_cubes = cubes.copy()
            for nyears, cube in enumerate(new_cubes):
                add_years_onto_time(cube, nyears = nyears - 8)
        
            final_cube = new_cubes.concatenate_cube() 
        except:
            set_trace()
            cube = cubes[-1]
            time_coord = cube.coord('time')
            cube.remove_coord('time')
            cube = iris.util.new_axis(cube, time_coord)
    return final_cube

def make_input(variable, region, dir, eg_file, start_year = 2002):
    
    out_name, in_file, f_dir, cf_dir, varname, FUN = variable

    region = region.replace(' ', '_')
    eg_file = eg_file.replace('REGION_NAME', region)
    eg_cube = iris.load_cube(eg_file)
    target_time_unit = eg_cube.coord('valid_time').units
    eg_cube = eg_cube[0]

    if f_dir[0] == '.' or f_dir[0] == '~' or f_dir[0] == '/':
        filename = f_dir + '/' + in_file
    else:
        filename = dir + region +'/' + f_dir + '/' + in_file
    
    files = sorted(glob.glob(filename))
    
    cubes = iris.load(files, varname)
    
    if len(cubes) == 1: 
        cube = cubes[0]
    else:
        if cubes[0].ndim == 2:
            cube = combine_2d_cubes(cubes, files, target_time_unit)
        else:
            cube = combine_3d_cubes(cubes, region, files)
    time_coord = cube.coords()[0].name()
    
    try:
        icc.add_year(cube, time_coord)
    except:
        pass
    try:
        icc.add_month(cube, time_coord)
    except:
        pass
    
    cube = sub_year_range(cube, [start_year, 9999])
    cube = cube.aggregated_by(['year', 'month'], FUN)
    cube = regrid_to_eg_cube(cube, eg_cube)
    out_file = dir.replace('nrt_raw', region) + '/nrt/factual/' + out_name + '.nc'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    iris.save(cube, out_file)
    
if __name__=="__main__":
    dir = "data/data/driving_data2526/nrt_raw/"
    
    regions = ["Northwest Iberia", "Scottish Highlands"]
    
    region = regions[0]
    start_year = 2002
    eg_file = "data/data/driving_data2526/nrt_raw/REGION_NAME/ERA5_Factual/pr.nc"

    Joeys_data = "/data/users/douglas.kelley/Bayesian_fire_models/Joeys/SOW_FORCINGS/"
                #outname, inname, factual dir, count dir
    variables = [#["DFMC_Wood", "FUEL/DFMC_timemean_*.nc", Joeys_data, None, "DFMC_Wood",    
                 # iris.analysis.MEAN],
                 #["DFMC_Foliage", "FUEL/DFMC_timemean_*.nc", Joeys_data, None, "DFMC_Foliage",  
                 # iris.analysis.MEAN],
                 #["LAI_high", "VEG/month_laih*_C.nc", Joeys_data, None, None, 
                 # iris.analysis.MEAN],
                ["LAI_lowh", "VEG/month_lail*_C.nc", Joeys_data, None, None, 
                  iris.analysis.MEAN],
                ["cvh", "VEG/cvh*_BA.nc", Joeys_data, None, None, 
                  iris.analysis.MEAN],
                ["cvl", "VEG/cvl*_BA.nc", Joeys_data, None, None, 
                  iris.analysis.MEAN],
                ["tvl", "VEG/tvl*_BA.nc", Joeys_data, None, None, 
                  iris.analysis.MEAN],
                 ["LFMC_high", "FUEL/LFMC_timemean_*.nc", Joeys_data, None, "LFMC_high",  
                  iris.analysis.MEAN],
                 ["LFMC_low", "FUEL/LFMC_timemean_*.nc", Joeys_data, None, "LFMC_low",  
                  iris.analysis.MEAN],
                 #["LAI_low", "VEG/month_lail*_BA.nc", Joeys_data, None, None, 
                 # iris.analysis.MEAN],
                 #["tas", "tas.nc", "ERA5_Factual", "HadGEM_Counter", None,
                 # iris.analysis.MEAN]
                 ]
    for variable in variables:
        make_input(variable, region, dir, eg_file, start_year)

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
from datetime import date

import iris.quickplot as qplt
import matplotlib.pyplot as plt

def regrid_to_eg_cube(cube, eg_cube):
    if cube.ndim == 3:
        if cube.shape[1:] == eg_cube.shape: 
            return cube
        lat_name = cube.coords()[1].name()
        lon_name = cube.coords()[2].name()
    elif cube.ndim == 2:
        if cube.shape == eg_cube.shape: 
            return cube
        lat_name = cube.coords()[0].name()
        lon_name = cube.coords()[1].name()
    else:
        set_trace()
    cube.coord(lat_name).rename(eg_cube.coords()[0].name())
    cube.coord(lon_name).rename(eg_cube.coords()[1].name())
    
    cube.coord('latitude').units = eg_cube.coord('latitude').units
    cube.coord('longitude').units = eg_cube.coord('longitude').units
    cube.coord('latitude').coord_system = eg_cube.coord('latitude').coord_system
    cube.coord('longitude').coord_system = eg_cube.coord('longitude').coord_system
    
    out =  cube.regrid(eg_cube, iris.analysis.Linear())
    return out


def add_time_based_on_mnth_year(cubes, years, months, time_unit):
    unit = cf_units.Unit(time_unit, calendar='gregorian')
    
    datetimes = [
        datetime.datetime(y, m, 15)
        for y, m in zip(years, months)
    ]
    time_points = unit.date2num(datetimes)

    new_cubes = []
    try:
        ncubes = cubes.shape[0]
    except:
        ncubes = len(cubes)
        
    for i in range(ncubes):
        cube = cubes[i]
        
        t = time_points[i]
        
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
        
        try:
            cube.remove_coord('time')
        except:
            pass
        cube.add_aux_coord(time_coord)
        cube = iris.util.new_axis(cube, 'time')  # promote to dimension
        
        new_cubes.append(cube)
    iris.util.equalise_attributes(new_cubes)
    new_cubes = iris.cube.CubeList(new_cubes)
    try:
        new_cubes = new_cubes.concatenate_cube(new_cubes)
    except:
        for cube in new_cubes:
            cube.rename(new_cubes[0].name())
        new_cubes = new_cubes.concatenate_cube(new_cubes)
    return new_cubes
def combine_2d_cubes(cubes, files, time_unit):
    years = [int('20' + file.split('_20')[1][0:2]) for file in files]
    months = [int(file.split('_20')[1][3:5]) for file in files]
    #years = [int(file[-10:-6]) for file in files]
    #months = [int(file[-5:-3]) for file in files]
    return add_time_based_on_mnth_year(cubes, years, months, time_unit)
        
def combine_3d_cubes_blank_time(cube, file, time_unit):
    
    year = '20' + file.split('_20')[1][0:2]
    years = np.tile(int(year), cube.shape[0])
    
    months = np.arange(1, cube.shape[0]+1, dtype=int)
    
    return add_time_based_on_mnth_year (cube, years, months, time_unit)


def set_calendar(cube, target_time_unit):
    try:
        cube.coord('time').convert_units(target_time_unit)
    except:
        time_coord = cube.coord('time')
        
        datetimes = time_coord.units.num2date(time_coord.points)
        
        new_unit = cf_units.Unit('hours since 2003-01-15 00:00:00',
                                 calendar='proleptic_gregorian')
    
        time_coord.points = new_unit.date2num(datetimes)
        time_coord.units = new_unit
        cube.coord('time').convert_units(target_time_unit)
    return cube
    

def combine_3d_cubes(cubes, region, files):
    vname = cubes[0].name()
    
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
            try:
                cube = fix_joeys_weird_time_coords(cube, file)
            except:
                cube = combine_3d_cubes_blank_time(cube, file, target_time_unit)
                
        set_calendar(cube, target_time_unit)
        
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
        try:
            cube.units = new_cubes[0].units
        except:
            pass
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

def make_input(variable, region, dir, eg_file, start_year = 2002, shapefile_path = None):
    
    out_name, in_file, f_dir, cf_dir, varname, FUN = variable

    if not isinstance(FUN, list): 
        FUN = [None, FUN]

    region = region.replace(' ', '_')
    eg_file = eg_file.replace('REGION_NAME', region)
    eg_cube = iris.load_cube(eg_file)
    
    target_time_unit = eg_cube.coord('time').units
    eg_cube = eg_cube[0]
    
    def make_subout(i_dir, o_dir, counter = False):
        
        def make_file(files, ens = None):
            cubes = iris.load(files, varname)
            
            if len(cubes) == 1: 
                cube = cubes[0]
            else:
                if len(files[0].split('LI/LI_'))== 2:
                    new_cubes = []
                    for cube in cubes:
                        new_cubes.append(cube[0])
                    cubes = iris.cube.CubeList(new_cubes)
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
            
            if FUN[0] is not None:
                cube = FUN[0](cube)
            
            year_today = date.today().year
            if len(cube.shape) > 2:
                cube = sub_year_range(cube, [start_year, year_today])
                cube = cube.aggregated_by(['year', 'month'], FUN[1])
                icc.add_month_number(cube, 'time')
                index = np.any(np.array([cube.coord('month_number').points < 3,
                                         cube.coord('year').points < year_today]), 
                               axis = 0)
                cube = cube[index]
            cube = regrid_to_eg_cube(cube, eg_cube)
            
            if ens is not None:
                ens_txt = '/ens-' + str(ens)
            else:
                ens_txt = ''
            out_file = dir.replace('nrt_raw', region) + '/nrt/' + o_dir + '/' + \
                        out_name + ens_txt + '.nc'
            os.makedirs(os.path.dirname(out_file), exist_ok=True)  
            
            cube.data = np.nan_to_num(cube.data.filled(np.nan), nan=0)
            if shapefile_path is not None:
                cube = contrain_to_sow_shapefile(cube, shapefile_path, 
                                                 region.replace('_', ' '))
            
            iris.save(cube, out_file)

        sl = '/' if counter else ''
        if i_dir[0] == '.' or i_dir[0] == '~' or i_dir[0] == '/':
            filename = i_dir + '/' + in_file + sl + '*'
        else:
            
            if in_file == 'tas' and not counter:
                filename = dir + region +'/' + i_dir + '/' + in_file + '.nc'
            else:
                filename = dir + region +'/' + i_dir + '/' + in_file + sl + '*'
        #set_trace()
        files = sorted(glob.glob(filename, recursive = True))#[0:6]   
        
        if counter:
            [make_file(file, i) for i, file in enumerate(files)]
        else:
            make_file(files)
    
    make_subout(f_dir, 'factual')

    if cf_dir is not None:
        make_subout(cf_dir, 'countfactual', True)
    #set_trace()
    
    
    
def dry_day(cube):
    mask = cube.data < 0.0001
    cube.data[:] = 0
    cube.data[mask] = 1
    cube.rename('Dry days')
    cube.units = 'day'
    return cube

def cummulative_dry_day(cube):
    cube = dry_day(cube)
    cube_count = cube.data[0].copy()
    for i in range(1, cube.shape[0]):
        cube_count += 1
        mask = cube.data[i] == 0
        cube_count[mask] = 0
        cube.data[i] = cube_count

    cube.rename('Cummlative dry days')
    return cube

if __name__=="__main__":
    dir = "data/data/driving_data2526/nrt_raw/"
    
    shapefile_path = "data/data/driving_data2526/Focal_regions/SoW2526_Focal_MASTER_20260218.shp" 
    regions = [#"Midwestern Canadian Shield forests", 
               "Chilean Temperate Forests and Matorral", 
               #"Southeast South Korea", 
               #"Northwest Iberia", 
               "Scottish Highlands"
               ]
    
    region = regions[0]
    start_year = 2002
    eg_file = "data/data/driving_data2526/nrt_raw/REGION_NAME/ERA5_Factual/pr.nc"

    Joeys_data = "/data/users/douglas.kelley/Bayesian_fire_models/Joeys/SOW_FORCINGS/"
    BA_dir = "/home/users/douglas.kelley/Bayesian_fire_models/data/data"
    hadgem_veg_frac = "/home/users/douglas.kelley/" + \
                      "Bayesian_fire_models/data/data/HadGEM_land_frac/"
                #outname, inname, factual dir, count dir
    variables = [
                 ["tree_HYDE31", "tree", hadgem_veg_frac + "/factual", 
                  hadgem_veg_frac + "/counterfactual", None,
                  iris.analysis.MEAN],
                 ["wood_HYDE31", "wood", hadgem_veg_frac + "/factual", 
                  hadgem_veg_frac + "/counterfactual", None,
                  iris.analysis.MEAN],
                 ["veg_HYDE31", "veg_abs", hadgem_veg_frac + "/factual", 
                  hadgem_veg_frac + "/counterfactual", None,
                  iris.analysis.MEAN],
                 ["veg_HYDE31_log", "veg_log", hadgem_veg_frac + "/factual", 
                  hadgem_veg_frac + "/counterfactual", None,
                  iris.analysis.MEAN],
                 ["LI", "LI/LI_*C*", Joeys_data, None, "litoti",    
                  iris.analysis.MEAN],
                 ["burned_area", "burned_area_global.nc", BA_dir, "None", None,
                  iris.analysis.MEAN],
                 ["dry_days", "pr", "ERA5_Factual", "HadGEM_Counter", None,
                  [dry_day, iris.analysis.MEAN]],
                 ["cumm_dry_days_mean", "pr", "ERA5_Factual", "HadGEM_Counter", None,
                  [cummulative_dry_day, iris.analysis.MEAN]],
                 ["cumm_dry_days_max", "pr", "ERA5_Factual", "HadGEM_Counter", None,
                  [cummulative_dry_day, iris.analysis.MAX]],
                 ["tasmax", "tasmax", "ERA5_Factual", "HadGEM_Counter", None,
                  iris.analysis.MAX],
                 #["tas", "tas", "ERA5_Factual", "HadGEM_Counter", None,
                 # iris.analysis.MEAN],
                 ["pr", "pr", "ERA5_Factual", "HadGEM_Counter", None,
                  iris.analysis.MEAN],
                 ["wind_mean", "wind", "ERA5_Factual", "HadGEM_Counter", None,
                  iris.analysis.MEAN],
                 ["wind_max", "wind", "ERA5_Factual", "HadGEM_Counter", None,
                  iris.analysis.MAX],
                 ["gust1_mean", "WindGust1", "ERA5_Factual", "HadGEM_Counter", None,
                  iris.analysis.MEAN],
                 ["gust1_max", "WindGust1", "ERA5_Factual", "HadGEM_Counter", None,
                  iris.analysis.MAX],
                 ["gust2_mean", "WindGust2", "ERA5_Factual", "HadGEM_Counter", None,
                  iris.analysis.MEAN],
                 ["gust2_max", "WindGust2", "ERA5_Factual", "HadGEM_Counter", None,
                  iris.analysis.MAX],
                 ["hursmin", "hursmin", "ERA5_Factual", "HadGEM_Counter", None,
                  iris.analysis.MIN], 
                 ["DFMC_Wood", "FUEL/DFMC_timemean_", Joeys_data, None, "DFMC_Wood",    
                  iris.analysis.MEAN],
                 ["DFMC_Foliage", "FUEL/DFMC_timemean_", Joeys_data, None, "DFMC_Foliage",  
                  iris.analysis.MEAN],
                 ["LAI", "VEG/month_lai_", Joeys_data, None, None, 
                  iris.analysis.MEAN],
                 ["lle_pred", "FUEL/Fuel_pred_clip_", Joeys_data, None, "lle_pred", 
                  iris.analysis.MEAN],
                 ["lwo_pred", "FUEL/Fuel_pred_clip_", Joeys_data, None, "lwo_pred", 
                  iris.analysis.MEAN],
                 ["dfo_pred", "FUEL/Fuel_pred_clip_", Joeys_data, None, "dfo_pred", 
                  iris.analysis.MEAN],
                 ["dwo_pred", "FUEL/Fuel_pred_clip_", Joeys_data, None, "dwo_pred", 
                  iris.analysis.MEAN],
                 ["LFMC_high", "FUEL/LFMC_timemean_", Joeys_data, None, "LFMC_high",  
                  iris.analysis.MEAN],
                 ["LFMC_low", "FUEL/LFMC_timemean_", Joeys_data, None, "LFMC_low",  
                  iris.analysis.MEAN]
                 ]
    for variable in variables:
        make_input(variable, region, dir, eg_file, start_year, shapefile_path)

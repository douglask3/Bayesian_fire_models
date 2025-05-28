from pdb import set_trace
import os.path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
import sys
from pathlib import Path

sys.path.append('../../libs/')
sys.path.append('libs/')
from constrain_cubes_standard import *

from shapely.ops import unary_union
import shapely.vectorized

import iris
import iris.analysis

import cftime
import cf_units

from pprint import pprint

def print_diffs(cube1, cube2):
    print("== ATTRIBUTES ==")
    pprint({k: (cube1.attributes.get(k), cube2.attributes.get(k)) 
            for k in set(cube1.attributes) | set(cube2.attributes) 
            if cube1.attributes.get(k) != cube2.attributes.get(k)})

    print("\n== CELL METHODS ==")
    print(cube1.cell_methods)
    print(cube2.cell_methods)

    print("\n== NAMES ==")
    print("standard_name:", cube1.standard_name, cube2.standard_name)
    print("long_name:", cube1.long_name, cube2.long_name)
    print("var_name:", cube1.var_name, cube2.var_name)

    print("\n== AUX COORDS ==")
    print([coord.name() for coord in cube1.aux_coords])
    print([coord.name() for coord in cube2.aux_coords])


def process_variable(experiment, variable, start_year, dir, sub_dir, out_dir, temp_dir, 
                     region_names = None, shapefile_path = None):
    
    var_dir = dir  + '/' + experiment[0] + '/' + variable + '/' + sub_dir + '/'
    
    files = os.listdir(var_dir)
    
    files = [file for file in files if file[-3:] == '.nc']
    
    try:
        ensembles = [file[-18:-10] for file in files if file[-18] == 'r']
    except:
        set_trace()
    
    #shapes = gpd.read_file(shapefile_path)
    ensembles = list(set(ensembles))
    
    def process_memember(member):
        print(member)
        completed_file = temp_dir + experiment[1] + variable + member + str(start_year) + shapefile_path.replace('/', '-') + '.txt'
        
        #if os.path.isfile(completed_file) and False:
        #    return
        mfiles = [file for file in files if member in file]
        mfiles = [file for file in mfiles if int(file[-9:-5]) >= start_year]
        mfiles.sort()
        
        mfiles = [var_dir + file for file in mfiles]
        if len(mfiles) == 0:
            set_trace()
        
        cube = iris.load(mfiles)
        for cb in cube:
            cb.coord('time').bounds = None
        for i in range(len(cube)):
            cube[i].coord('time').bounds = None
            cube[i] = cube[i][0:90]
        iris.util.equalise_attributes(cube)
        iris.util.unify_time_units(cube)
        all_slices = []
        cubes = cube.copy()
        cube = cube.concatenate()        
        
        if len(cube) > 1:
            try:
                for cube in cubes:
                    for t in range(cube.shape[0]):
                        all_slices.append(cube[t])
                
                time_to_slice = {}
                for slc in all_slices:
                    time_val = slc.coord('time').points[0]
                    # Only keep the first occurrence of each time value
                    if time_val not in time_to_slice:
                        time_to_slice[time_val] = slc
                
                unique_times = sorted(time_to_slice.keys())
                unique_slices = [time_to_slice[t] for t in unique_times]
                
                cube = iris.cube.CubeList(unique_slices).merge_cube()
                '''
                print_diffs(cube[0], cube[1])
                set_trace()
                for i, cb in enumerate(cube):
                    print(f"--- Cube {i} ---")
                    for coord in cb.coords():
                        print(f"{coord.name()} ({coord.shape}) units={coord.units} bounds={coord.bounds is not None}")
                '''
            except:
                return None
        else:
            cube = cube[0]
        print("yay")
        def process_region(region_name, cube):
            out_file = out_dir + '/' + region_name.replace(' ', '_') + \
                       '/HadGEM_' + experiment[1] + \
                       '/' + variable + '/' + member + '-' + str(start_year) + '-2.nc'
            #if os.path.isfile(out_file):
            #    return
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            
            cube.coord("longitude").circular = True
            cube = cube.intersection(longitude=(-180, 180))
            cube = contrain_to_sow_shapefile(cube, shapefile_path, region_name)
            
            iris.save(cube, out_file, local_keys=['calendar'])
        [process_region(region_name, cube) for region_name in region_names]

        os.makedirs(os.path.dirname(completed_file), exist_ok=True)
        Path(completed_file).touch()
    [process_memember(member) for member in ensembles]
    


def process_variables(experiments, variables, *args, **kw):
    for experiment in experiments:
        print(experiment)
        for variable in variables:
            process_variable(experiment, variable, *args, **kw)
    
if __name__=="__main__":
    dir = "/data/users/opatt/HadGEM3-A-N216/"
    sub_dir = '/day/'

    temp_dir = "/data/scratch/douglas.kelley/Bayesian_fire_models/temp/hadgem_nrt/"
    out_dir = "data/data/driving_data2425/nrt_attribution/"

    start_years = [2023, 2013]
    start_years = [2013]
    shapefile_path = "data/data/SoW2425_shapes/SoW2425_Focal_MASTER_20250221.shp"
    region_names = ["northeast India",
                   "Alberta",
                   "Los Angeles",
                   "Congo basin",
                   "Amazon and Rio Negro rivers",
                   "Pantanal basin"]

    variables = ['pr', 'tasmax']
    variables = ['tasmax']
    variables = ['hursmin']#, 'tas']#, 
    variables = ['sfcWind']#, 'uas', 'vas',  'mrros', 
    experiments = [['historicalNatExt', 'NAT'], ['historicalExt', 'ALL']]
    for start_year in start_years:
        process_variables(experiments, variables, start_year, dir, sub_dir,
                          out_dir, 
                          temp_dir,
                          region_names = region_names,
                          shapefile_path = shapefile_path)



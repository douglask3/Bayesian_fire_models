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
        #set_trace()
        if os.path.isfile(completed_file) and False:
            return
        mfiles = [file for file in files if member in file]
        mfiles = [file for file in mfiles if int(file[-9:-5]) >= start_year]
        mfiles.sort()
        mfiles = [var_dir + file for file in mfiles]
        if len(mfiles) == 0:
            set_trace()

        cube = iris.load(mfiles).concatenate()
        if len(cube) > 1:
            return
        else:
            cube = cube[0]
        
        def process_region(region_name, cube):
            out_file = out_dir + '/' + region_name.replace(' ', '_') + \
                       '/HadGEM_' + experiment[1] + \
                       '/' + variable + '/' + member + '-' + str(start_year) + '.nc'
            if os.path.isfile(out_file):
                return
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            
            
            cube.coord("longitude").circular = True
            cube = cube.intersection(longitude=(-180, 180))
            cube = contrain_to_sow_shapefile(cube, shapefile_path, region_name)

            iris.save(cube, out_file)
            
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

    start_year = 2023

    shapefile_path = "data/data/SoW2425_shapes/SoW2425_Focal_MASTER_20250221.shp"
    region_names = ["northeast India",
                   "Alberta",
                   "Los Angeles",
                   "Congo basin",
                   "Amazon and Rio Negro rivers",
                   "Pantanal basin"]

    variables = ['tasmax', 'tas', 'pr', 'hursmin', 'sfcWind']#, 'uas', 'vas', 
    experiments = [['historicalExt', 'ALL'], ['historicalNatExt', 'NAT']]
    process_variables(experiments, variables, start_year, dir, sub_dir,
                     out_dir, 
                     temp_dir,
                     region_names = region_names,
                     shapefile_path = shapefile_path)



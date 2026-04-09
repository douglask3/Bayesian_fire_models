from pdb import set_trace
import os.path
import os
from pathlib import Path


import numpy as np
import iris

import iris.quickplot as qplt
import matplotlib.pyplot as plt
from pdb import set_trace

import sys

sys.path.append('libs/')
from constrain_cubes_standard import *

def make_humid(inputs):
    def vp(cube):
        cube.data =  6.112 * np.exp((17.625 * cube.data)/(cube.data + 243.03))
        return cube 
    
    def humid(syr, dyr):
        print(syr)
        eyr = syr + dyr
        return vp(inputs[0][syr:eyr])/vp(inputs[1][syr:eyr])   
        
    out = [humid(i, min(50, inputs[0].shape[0]-i)) for i in range(0, inputs[0].shape[0], 50)]
    return iris.cube.CubeList(out).concatenate_cube()
    

def make_extra_var(variables, FUN, out_name, region, dir, experiment, file):
    def open_var(var):
        filename = dir + region.replace(' ', '_')  + '/'  + experiment + '/' + var + '/' + file
        cube = iris.load_cube(filename)
        cube = contrain_to_sow_shapefile(cube, "data/data/driving_data2526/Focal_regions/SoW2526_Focal_MASTER_20260218.shp", region)
        
        return cube

    out_filename = dir + region.replace(' ', '_')  + '/'  + \
                    experiment + '/' + out_name + '/' + file

    if Path(out_filename).exists(): 
        return None
    inputs = [open_var(var) for var in variables]
    inputs = constrain_to_common_time(inputs)
    
    output = FUN(inputs)
    Path(out_filename).parent.mkdir(parents=True, exist_ok=True)
    iris.save(output, out_filename)

def make_era5_extra_vars(dir, regions, 
                         experiment = "Era5_derived-era5-single-levels-daily-statistics/",
                         files = ['_years2024-20262.nc'],
                         variables = [["tasdew", "tasmax"], ["tasdew", "tas"]],
                         FUNs = [make_humid, make_humid],
                         output_names = ['hursmin', 'hurs']):
    for region in regions:
        for file in files:
            for vars, FUN, outname in zip(variables, FUNs, output_names):
                make_extra_var(vars, FUN, outname, region, dir, experiment, file)

if __name__=="__main__":
    dir = "data/data/driving_data2526/nrt_raw/"
    
    regions = ["Northwest Iberia", "Scottish Highlands"]
    
    make_era5_extra_vars(dir, regions)
    

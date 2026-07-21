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

def make_vpd(inputs):
    T = inputs[0]
    rh =inputs[1]
    
    def vpd(syr, dyr):
        print(syr)
        eyr = syr + dyr
        out = inputs[0][syr:eyr].copy()
        vp_leaf = 0.61078 * np.exp((17.27 * T[syr:eyr].data) / (T[syr:eyr].data + 237.3))
        
        out.data = vp_leaf - vp_leaf * rh[syr:eyr].data / 100.0
        return out
    out = [vpd(i, min(50, inputs[0].shape[0]-i)) for i in range(0, inputs[0].shape[0], 50)]
    
    return iris.cube.CubeList(out).concatenate_cube()
    

def make_extra_var(variables, FUN, out_name, region, dir, experiment, file_ext):
        
        
    def for_file(file):
        out_filename = dir + region.replace(' ', '_')  + '/'  + \
                        experiment + '/' + out_name + '/' + file

        def open_var(var):
            
            filename = dir + region.replace(' ', '_')  + '/'  + experiment + '/' + \
                        var + '/' + file
            cube = iris.load_cube(filename)
            cube = contrain_to_sow_shapefile(cube, "data/data/driving_data2526/Focal_regions/SoW2526_Focal_MASTER_20260218.shp", region)

            return cube
        #if Path(out_filename).exists(): 
        #    return None
        
        inputs = [open_var(var) for var in variables]
        inputs = constrain_to_common_time(inputs)
        
        output = FUN(inputs)
        Path(out_filename).parent.mkdir(parents=True, exist_ok=True)
        iris.save(output, out_filename)

    dirs = [dir + region.replace(' ', '_')  + '/'  + experiment + '/' + var + '/'\
            for var in variables]
    
    file_sets = [
        {f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))}
        for d in dirs
    ]
    common_files = set.intersection(*file_sets)
    [for_file(file) for file in common_files]
    
    


def make_HadGEM_extra_vars(dir, regions, 
                         experiments = ["HadGEM_ALL", "HadGEM_NAT"],
                         file_ext = "-2023-2.nc",
                         variables = [["tasmax", "hursmin"]],
                         FUNs = [make_vpd],
                         output_names = ['vpd']):
    for region in regions:
        for experiment in experiments:
            for vars, FUN, outname in zip(variables, FUNs, output_names):
                make_extra_var(vars, FUN, outname, region, dir, experiment, file_ext)



if __name__=="__main__":
    dir = "data/data/driving_data2526/nrt_raw/"
    
    regions = [
               #"Midwestern Canadian Shield forests", 
               #"Chilean Temperate Forests and Matorral", 
               #"Southeast South Korea", 
               "Northwest Iberia", 
               "Scottish Highlands"
               ]
    
    make_HadGEM_extra_vars(dir, regions)


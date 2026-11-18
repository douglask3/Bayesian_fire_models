from pdb import set_trace
import os.path
import os
from pathlib import Path
import glob

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
        Tc = cube.data - 273.15
        cube.data =  6.112 * np.exp((17.625 * Tc)/(Tc + 243.03))
        return cube 
    
    def humid(syr, dyr):
        print(syr)
        eyr = syr + dyr
        return vp(inputs[0][syr:eyr])/vp(inputs[1][syr:eyr])   
        
    out = [humid(i, min(50, inputs[0].shape[0]-i)) for i in range(0, inputs[0].shape[0], 50)]
    return iris.cube.CubeList(out).concatenate_cube()

def make_vpd(inputs):
    T = inputs[1]
    rh = make_humid(inputs)
    
    
    def vpd(syr, dyr):
        print(syr)
        eyr = syr + dyr
        out = inputs[0][syr:eyr].copy()
        Tc = T[syr:eyr].data -273.15
        vp_leaf = 610.78 * np.exp((17.27 * Tc) / (Tc + 237.3))
        out.data = vp_leaf - vp_leaf * rh[syr:eyr].data
        
        return out
    out = [vpd(i, min(50, inputs[0].shape[0]-i)) for i in range(0, inputs[0].shape[0], 50)]
    
    return iris.cube.CubeList(out).concatenate_cube()
    

def make_wind(inputs):
    out = inputs[0].copy()
    out.data = np.sqrt(inputs[0].data**2 + inputs[1].data**2) 
    out.rename('wind speed')
    return out
    
def make_extra_var(variables, FUN, out_name, region, dir, experiment, file, shapeinfo):
    def open_var(var):
        filename = dir + region.replace(' ', '_')  + '/'  + experiment + '/' + var + '/' + file
        
        filenames = glob.glob(filename)
        def open_constrain(filename):
            cube = iris.load_cube(filename)
            if shapeinfo is not None:
                for shpinf in shapeinfo:
                    cube = shpinf[0](cube, **shpinf[1])
            return cube, filename
        return [open_constrain(filename) for filename in filenames]

    
    
    
    inputs = [open_var(var) for var in variables]
    for i in range(len(inputs[0])):
        
        out_filename = dir + region.replace(' ', '_')  + '/'  + \
                    experiment + '/' + out_name + '/' + inputs[0][i][1].split('/')[-1]
        if Path(out_filename).exists(): 
            continue
        input = [cubes[i][0] for cubes in inputs]
        input = constrain_to_common_time(input)
        
        output = FUN(input)
        Path(out_filename).parent.mkdir(parents=True, exist_ok=True)
        iris.save(output, out_filename)

def make_era5_extra_vars(dir, regions, shapeinfo,
                         experiment = "Era5_derived-era5-single-levels-daily-statistics/",
                         files = ['_years*.nc'],
                         variables = [["u-wind", "v-wind"], 
                                      ["tasdew", "tasmax"], 
                                      ["tasdew", "tas"], 
                                      ["tasdew", "tasmax"]],
                         FUNs = [make_wind, make_humid, make_humid, make_vpd],
                         output_names = ['wind', 'hursmin', 'hurs', 'vpd']):
    for region in regions:
        for file in files:
            for vars, FUN, outname in zip(variables, FUNs, output_names):
                make_extra_var(vars, FUN, outname, region, dir, experiment, file, shapeinfo)

if __name__=="__main__":
    dir = "data/data/driving_data2526/nrt_raw/"
    
    regions = [
               "Midwestern Canadian Shield forests", 
               "Chilean Temperate Forests and Matorral", 
               #"Southeast South Korea", 
               #"Northwest Iberia", 
               #"Scottish Highlands"
               ]
    
    make_era5_extra_vars(dir, regions)
    

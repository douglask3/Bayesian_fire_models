import iris
import numpy as np
import cftime
from iris.analysis import Nearest, Linear
from pdb import set_trace
from pathlib import Path
import iris.quickplot as qplt
import matplotlib.pyplot as plt

def find_monthly_change(region, variable, target_dir, target_subdir):
    dir = target_dir + region + target_subdir
    files = Path(dir).iterdir()

    files = [f.name for f in files if f.is_file() and variable in f.name]
    files = [file for file in files if 'change_' not in file]
    
    def change_in_file(file):
        cube = iris.load_cube(dir + file)
        cube_out = cube[1:]
        cube_out.data -= cube.data[:-1]
        cube_out.data *= -1
        cube_out.data[cube_out.data<0.0] = 0.0
        iris.save(cube_out, dir + 'change_' + file )
        
    [change_in_file(file) for file in files]
    

if __name__=="__main__":
    regions = ["Pantanal", "Amazon"]
    target_dir = "data/data/driving_data_base/"
    target_subdir = "/nrt/era5_monthly/"
    
    variables = ["Wetland", "Forest"]
    
    for variable in variables:
        for region in regions:
            #find_monthly_change(region, variable, target_dir, target_subdir)
            try:
                find_monthly_change(region, variable, target_dir, target_subdir)
            except:
                print("No change file generated for "  + variable + \
                      " in region " + region + ".")

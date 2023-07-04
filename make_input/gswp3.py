import sys
sys.path.append('../libs/')
sys.path.append('libs/')

from read_variable_from_netcdf import *
from plot_maps import *
import os

from   io     import StringIO
import numpy  as np
import math

import matplotlib.pyplot as plt
from pdb import set_trace

def read_variable_from_netcdf_stack(filenames, example_cube = None, 
                                    *args, **kw):
    
    cubes = [read_variable_from_netcdf(file, *args, **kw) for file in filenames]
    cubes = [cube for cube in cubes if cube is not None]
    cubes = iris.cube.CubeList(cubes)
    
    cubes = cubes.concatenate_cube()
    
    if example_cube is not None:
        example_cube = iris.load_cube(example_cube)
        
        cubes = cubes.regrid(example_cube, iris.analysis.Linear())
        cubes.data = np.ma.masked_greater(cubes.data, 9E9)
        cubes.data[cubes.data > 9E9] = np.nan
    return cubes


if __name__=="__main__":
    dir = "/hpc//data/d00/hadea/isimip3a/InputData/climate/atmosphere/obsclim/GSWP3-W5E5/gswp3-w5e5_obsclimfill_"
    
    file_years = ["1991_2000", "2001_2010", "2011_2019"]
    filenames = {"tas": "tas_global_daily_",
                 "tas_range": "tas_range_global_daily_",
                 "pr": "pr_global_daily_"}

    years = [[2000, 2009], [2010, 2019]]

    subset_functions = [constrain_natural_earth]
    subset_function_argss = [{'Country': 'Brazil'}]

    output_dir = "../ConFIRE_attribute/isimip3a/driving_data/GSWP3-W5E5-20yrs/Brazil/AllConFire_"
    output_years = ['2000_2009', '2010_2019']

    example_cube = '../ConFIRE_attribute/isimip3a/driving_data/GSWP3-W5E5-20yrs/Brazil/AllConFire_2000_2009/GFED4.1s_Burned_Fraction.nc'


    process = ['tas', 'pr']
    #process = ['pr']



    def make_variables_for_year_range(year, output_year):
        def test_if_process(var): return any(i == var for i in process)

        def open_variable(varname, plusMinusYr = False):
            filename = filenames[varname]
            filename = [filename + ext + '.nc' for ext in file_years]
            yeari = year.copy()
            if plusMinusYr:
                yeari[0] = yeari[0] - 1
                yeari[1] = yeari[1] + 1
            sbs_funs = [sub_year_range] + subset_functions 
            sbs_args = [{'year_range': yeari}] + subset_function_argss
            
            return read_variable_from_netcdf_stack(filename, example_cube, dir,
                                                   subset_function = sbs_funs, 
                                                   subset_function_args = sbs_args)

        def monthly_mean(cube, fun = iris.analysis.MEAN):
            return cube.aggregated_by(['year', 'month'], fun)

        def save_ncdf(cube, varname):
            iris.save(cube, output_dir + output_year + '/' + varname +  '.nc')

        if test_if_process('tas'):
            tas = open_variable('tas')
            tas_range = open_variable('tas_range')

            tas_monthly = monthly_mean(tas)

            tas_max = tas.copy()
            tas_max.data  = tas_max.data + 0.5 * tas_range.data
            tas_max_monthly = tas.aggregated_by(['year', 'month'], iris.analysis.MAX)

            save_ncdf(tas_monthly, 'tas_mean')
            save_ncdf(tas_max_monthly, 'tas_max')

        if test_if_process('pr'):
            pr = open_variable('pr', True)
            pr_mean = monthly_mean(sub_year_range(pr, year))
            
            dry_days = pr.copy()
            dry_days.data[dry_days.data < (1.0/86400.0)] = 0.0
            dry_days.data[dry_days.data > 0.0] = 1.0
            dry_days.data = 1.0 - dry_days.data
            dry_days.units = '1'

            def cummDry(data):
                if data.mask[0]: 
                    return data
                out = data.copy()
                out[:] = 0.0
                for i in range(1, len(out)):
                    if data[i] > 0.0: out[i] = out[i-1] + 1
                return out

            consec_dry = dry_days.copy()
            consec_dry.data = np.apply_along_axis(cummDry, 0, dry_days.data)
            
            dry_days_mean = monthly_mean(sub_year_range(dry_days, year))
            consec_dry_mean = monthly_mean(sub_year_range(consec_dry, year), iris.analysis.MAX)
            
            save_ncdf(pr_mean, 'pr_mean')
            save_ncdf(dry_days_mean, 'dry_days')
            save_ncdf(consec_dry_mean, 'consec_dry_mean')
        
    [make_variables_for_year_range(year, output_year) for year, output_year in \
            zip(years, output_years)]


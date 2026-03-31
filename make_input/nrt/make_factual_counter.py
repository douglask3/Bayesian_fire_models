from pdb import set_trace
import os.path
import os

import numpy as np

import iris
import iris.analysis
import cftime
import cf_units

import random
import itertools


import iris.quickplot as qplt
import matplotlib.pyplot as plt


def interplate_hadgem_to_era5_time(era5_cut, hadgem_cut, era5_time = 'valid_time', hadgem_time = 'time'):

    # Original cubes
    era5_coord = era5_cut.coord('valid_time')
    hadgem_coord = hadgem_cut.coord('time')

    # Convert ERA5 points to cftime.Datetime360Day using HadGEM calendar
    target_dates_360 = [cftime.Datetime360Day(d.year, d.month, min(d.day, 30))
                        for d in era5_coord.units.num2date(era5_coord.points)]

    # Interpolate HadGEM onto these target dates
    hadgem_interp = hadgem_cut.interpolate(
        [('time', [hadgem_coord.units.date2num(d) for d in target_dates_360])],
        iris.analysis.Linear()
    )
    
    # Rename coordinate to match ERA5
    hadgem_interp.coord('time').rename('valid_time')
    return hadgem_interp

def cut_era5_hadgem_to_time(era5, hadgem, era5_time = 'valid_time', hadgem_time = 'time'):
        
    # Extract coords
    era5_time = era5.coord(era5_time)
    hadgem_time = hadgem.coord(hadgem_time)

    # Convert to datetime-like objects
    era5_dates = era5_time.units.num2date(era5_time.points)
    hadgem_dates = hadgem_time.units.num2date(hadgem_time.points)
    
    # Convert to (year, month, day) tuples (ignore time-of-day)
    era5_ymd = np.array([(d.year, d.month, d.day) for d in era5_dates])
    hadgem_ymd = np.array([(d.year, d.month, d.day) for d in hadgem_dates])
    
    # Convert to set for intersection
    era5_set = set(map(tuple, era5_ymd))
    hadgem_set = set(map(tuple, hadgem_ymd))
    
    common_dates = sorted(era5_set & hadgem_set)
    
    # Get min/max common range
    start = common_dates[0]
    end = common_dates[-1]

    def in_range(ymd, start, end):
        return (ymd >= start) & (ymd <= end)
    
    # Boolean masks
    era5_mask = np.array([in_range(tuple(d), start, end) for d in era5_ymd])
    hadgem_mask = np.array([in_range(tuple(d), start, end) for d in hadgem_ymd])

    # Apply slicing
    era5_cut = era5[era5_mask]
    hadgem_cut = hadgem[hadgem_mask]
    
    return era5_cut, hadgem_cut


def crop_hadgem_era5_spatial_grids(era5, hadgem):
    lat_min = max(era5.coord('latitude').points.min(), hadgem.coord('latitude').points.min())
    lat_max = min(era5.coord('latitude').points.max(), hadgem.coord('latitude').points.max())
    lon_min = max(era5.coord('longitude').points.min(), hadgem.coord('longitude').points.min())
    lon_max = min(era5.coord('longitude').points.max(), hadgem.coord('longitude').points.max())
    lat_constraint = iris.Constraint(latitude=lambda v: lat_min <= v <= lat_max)
    lon_constraint = iris.Constraint(longitude=lambda v: lon_min <= v <= lon_max)

    era5_cropped = era5.extract(lat_constraint & lon_constraint)
    hadgem_cropped = hadgem.extract(lat_constraint & lon_constraint)

    lat_target = era5_cropped.coord('latitude').points
    lon_target = era5_cropped.coord('longitude').points
    
    # Interpolate HadGEM onto ERA5 grid
    hadgem_interp_spatial = hadgem_cropped.interpolate(
        [('latitude', lat_target), ('longitude', lon_target)],
        iris.analysis.Linear()
    )
    return era5_cropped, hadgem_interp_spatial

def make_variable_inputs(variable, transformation, inverse, datadir, regions, model_dir, 
                         experiments, obs_dataset, hadgem_start_year = 2019, npairs = 10):

    def make_region_input(region):
        
        def exp_files(experiment):
            dir = datadir + region.replace(' ', '_')  + '/'  + '/HadGEM_' + \
                experiments[0] + '/' + variable + '/'
            
            files = [dir + file for file in  os.listdir(dir)]
            files = [file for file in files if str(hadgem_start_year) in file.split('/')[-1]]
            return files
        file_lists = [exp_files(experiment) for experiment in experiments]
        
        all_pairs = list(itertools.product(file_lists[0], file_lists[1]))
        # Sample N unique pairs with replacement
        exp_files = set()
        
        while len(exp_files) < npairs:
            exp_files.add(random.choice(all_pairs))
        #set_trace() 
        exp_files = list(exp_files)
        
        dir = datadir + region.replace(' ', '_')  + '/'  + obs_dataset + '/' + variable + '/'
        files = os.listdir(dir)
        if len(files) > 1:
            years = np.array([int(file.split('_years')[1][0:4]) for file in files])
            files = files[np.argmin(years)]#set_trace()
        else:   
            files = files[0]
        obs_file = dir + files
        def open_data(file):
            cube = iris.load_cube(file)
            if transformation is not None: 
                cube.data = transformation(cube.data)
            return cube
    
        era5 = open_data(obs_file)
        for i, exp_file in enumerate(exp_files):
            #set_trace()
            ALL = open_data(exp_file[0])
            correct = open_data(exp_file[1])
            correct.data = correct.data - ALL.data
            era5, correct = cut_era5_hadgem_to_time(era5, correct)
            correct = interplate_hadgem_to_era5_time(era5, correct)
            era5, correct = crop_hadgem_era5_spatial_grids(era5, correct)
            cf = era5.copy()
            cf.data += correct.data 
            if inverse is not None:
                cf.data = inverse(cf.data)
            factual_file = exp_file[0].replace(experiments[0], 'Factual')
            factual_file = factual_file.replace('HadGEM', 'ERA5')
            factual_file = '/'.join(factual_file.split('/')[:-1]) + '.nc'
            counter_file = exp_file[0].replace(experiments[0], 'Counter')
            counter_file = '/'.join(counter_file.split('/')[:-1]) + '/ens-' + str(i)  + '.nc'
            if os.path.isfile(factual_file):
                os.makedirs(os.path.dirname(factual_file), exist_ok=True)
                iris.save(era5, factual_file)

            os.makedirs(os.path.dirname(counter_file), exist_ok=True)
            iris.save(cf, counter_file)
            
        
    [make_region_input(region) for region in regions]
    

model_dir = "/hadgem_nrt/"
obs_dataset = "/Era5_derived-era5-single-levels-daily-statistics/"

variables_obs = ['tasmax', 'tas', 'pr']
variables_mod = ['tasmax', 'tas', 'pr']
variables_out = ['tasmax', 'tax', 'pr']

def log1(x):
    return np.log(np.exp(x) -1)

def exp1(y):
    return np.log(np.exp(y) + 1)

transformations = [None, None, log1]
inverses = [None, None, exp1]

def make_all_variable_inputs(variables_obs, variables_mod, variables_out, 
                             transformations, inverses, *args, **kw):

    for vobs, vmod, vout, tran, invr in zip(variable_obs, variable_mod, 
                                            variable_out, transformations, inverses): 
        make_variable_inputs(vobs, vmod, vout, tran, invr, *args, **kw)

if __name__=="__main__":
    dir = "data/data/driving_data2526/nrt_raw/"

    experiments = ["ALL", "NAT"]
    regions = ["Scottish Highlands"]

    make_variable_inputs(variable_obs, variable_mod, variable_out, 
    transformations, inverses, dir,
                         regions, model_dir, 
                         experiments, obs_dataset)
"

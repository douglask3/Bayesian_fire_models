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


def make_variable_inputs(variable, transformation, inverse, datadir, regions, model_dir, 
                         experiments, obs_dataset, npairs = 10):

    def make_region_input(region):
        
        def exp_files(experiment):
            dir = datadir + region.replace(' ', '_')  + '/'  + '/HadGEM_' + \
                experiments[0] + '/' + variable + '/'
            
            return [dir + file for file in  os.listdir(dir)]
            
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
    
        obs_cube = open_data(obs_file)
        for exp_file in exp_files:
            set_trace()
            ALL = open_data(exp_file[0])
            corect = open_data(exp_file[1])
            corect.data = corect.data - ALL
            
            
        set_trace()
        
    [make_region_input(region) for region in regions]
    set_trace()

model_dir = "/hadgem_nrt/"
obs_dataset = "/Era5_derived-era5-single-levels-daily-statistics/"

variables = ['tasmax', 'tas', 'pr']
transformations = [None, None, np.log]
inverses = [None, None, np.exp]

if __name__=="__main__":
    dir = "data/data/driving_data2526/nrt_raw/"

    experiments = ["ALL", "NAT"]
    regions = ["Northwest Iberia"]

    make_variable_inputs(variables[0], transformations[0], inverses[0], dir,
                         regions, model_dir, 
                         experiments, obs_dataset)


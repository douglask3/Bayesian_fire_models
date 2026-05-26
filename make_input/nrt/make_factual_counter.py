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


def make_variable_inputs(variable, transformation, inverse, regions, model_dir, obs_dir, 
                         experiments, obs_dataset, npairs = 1000):

    def make_region_input(region):
        def exp_files(experiment):
            dir = model_dir + '/' + region + '/' + experiment + '/' + variable + '/'
            return [dir + file for file in  os.listdir(dir)]
            
        
        file_lists = [exp_files(experiment) for experiment in experiments]
        
        all_pairs = list(itertools.product(file_lists[0], file_lists[1]))
        # Sample N unique pairs with replacement
        exp_files = set()
        while len(exp_files) < 1000:
            exp_files.add(random.choice(all_pairs))
        
        exp_files = list(exp_files)
        
        dir = obs_dir + '/' + region + '/' + obs_dataset + '/' + variable + '/'
        files = os.listdir(dir)
        if len(files) == 1:
            obs_file = dir + files[0]
        else:
            set_trace()

        def open_data(file):
            cube = iris.load_cube(obs_file)
            if transformation is not None: 
                cube.data = transformation(cube.data)
            return cube
    
        obs_cube = open_data(obs_file)
        for exp_file in exp_files:
            
            ALL = open_data(exp_file[0])
            corect = open_data(exp_file[1])
            corect.data = corect.data - ALL
            set_trace()
            
        set_trace()
        
    [make_region_input(region) for region in regions]
    set_trace()

    



if __name__=="__main__":
    model_dir = "data/data/driving_data2425/hadgem_nrt/"
    obs_dir = "data/data/driving_data2425/era5_nrt/"

    experiments = ["ALL", "ALL"]
    obs_dataset = "derived-era5-single-levels-daily-statistics/"
    regions = ["Los_Angeles"]

    variables = ['tasmax', 'tas', 'pr']

    transformations = [None, None, np.log]
    inverses = [None, None, np.exp]

    make_variable_inputs(variables[0], transformations[0], inverses[0], 
                         regions, model_dir, obs_dir, 
                         experiments, obs_dataset)


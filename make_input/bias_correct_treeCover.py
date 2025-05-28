import iris
import numpy as np
import os
from pdb import set_trace


import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.pyplot as plt

def list_directories(root_dir):
    directories = []
    for root, dirs, files in os.walk(root_dir):
        for directory in dirs:
            directories.append(os.path.join(root, directory))
    return directories

def logit_0(x):
    scale = 1/np.prod(x.shape)    
    x = x * (1 - scale*2) + scale
    return np.log(x/(1-x))

def logistic(x):
    return 1.0/(1.0 + np.exp(-x))

def bias_correct_field(dir, file_original, cubet, start_target):
    file_original = dir + '/' + file_original

    cube0 = iris.load_cube(file_original)
    times0 = 1970 + cube0.coord('time').points/365.24
    timest = start_target + np.arange(0, cubet.shape[0])/12
    
    def mean_over_common(cube, time1, time2):
        which = np.where((time1 > time2[0]) & (time1 < time2[-1]))
        try:
            out = cube[which].collapsed('time', iris.analysis.MEAN)
        except:
            out = cube[which].collapsed('z', iris.analysis.MEAN)/100.0
        return out
    #which0 = np.where((times0 > timest[0]) & (times0 < timest[-1]))
    #whicht = np.where((timest > times0[0]) & (timest < times0[-1]))
    cube0_mean = mean_over_common(cube0, times0, timest)
    cubet_mean = mean_over_common(cubet, timest, times0)
    cubet_mean = cubet_mean.regrid(cube0_mean, iris.analysis.Linear())
    
    

    cube0_mean.data = logit_0(cubet_mean.data) - logit_0(cube0_mean.data)
    
    return cube0_mean

def bias_correct(dir, file_original, bias_cube, file_out):
    try:
        file_original = dir + '/' + file_original
        cube0 = iris.load_cube(file_original)
        
        cube1 = cube0.copy()
        cube1.data = logistic(bias_cube.data + logit_0(cube0.data))

        file_out = dir + '/' + file_out
        iris.save(cube1, file_out)
        print(file_out)
        return cube1
    except:
        return None

def bias_correct_veg(dir_for_correcting, dir_original, file_original, file_out, 
                     tcube, start_target):
    bias_cube = bias_correct_field(dir_original, file_original, cubet, start_target)
    for dirC in dir_for_correcting:
        dirs = list_directories(dirC)
        [bias_correct(dir, file_original, bias_cube, file_out) for dir in dirs]
    

dir_for_correcting = ['/data/dynamic/dkelley/ConFIRE_ISIMIP/isimip3a/driving_data/ERA5/Canada/',
                      '/scratch/dkelley/ConFire/inputs/isimip3b/Canada/']
dir_original = '/data/dynamic/dkelley/ConFIRE_ISIMIP/isimip3a/driving_data/ERA5/Canada/historic_TS_2001_2020/obsclim/'

file_original = 'trees.nc'
file_out = 'tree_biascorrected.nc'
target_file = '/home/h02/dkelley/fireMIPbenchmarking/data/benchmarkData/treecover2000-2014.nc'

start_target = 2000.5
cubet = iris.load_cube(target_file)


    
bias_correct_veg(dir_for_correcting, dir_original, file_original, file_out, cubet, start_target)

file_original = 'totalVeg.nc'
file_out = 'totalVeg_biascorrected.nc'
target_file = '/home/h02/dkelley/fireMIPbenchmarking/data/benchmarkData/bareground2000-2014.nc'

cubet = iris.load_cube(target_file)
cubet.data = 100.0 - cubet.data
bias_correct_veg(dir_for_correcting, dir_original, file_original, file_out, cubet, start_target)


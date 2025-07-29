import iris
import iris.coord_categorisation as icc
import numpy as np
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import datetime
import os
import pandas as pd
import math
import sys
sys.path.append('libs/')
from constrain_cubes_standard import *
from climtatology_difference import *

from pdb import set_trace

def for_region(region, i, fig, axes, time = None, filename_extra = '', csv_dir_out = ''):
    # Load the data
    cube = iris.load_cube("data/data/driving_data2425/" + region + "/burnt_area.nc")
    
    
    if time is not None:
        cube = sub_year_months(cube, time)
        icc.add_year(cube, 'time')
        cube = cube.aggregated_by('year', iris.analysis.MEAN)
        nmnths = len(time)
    else:
        nmnths = 12
        
    
    # Extract time coordinate
    time_coord = cube.coord('time') 
    time_points = [datetime.datetime(t.year, t.month, t.day) \
                   for t in time_coord.units.num2date(time_coord.points)]
    
    
    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()
    
    grid_areas = iris.analysis.cartography.area_weights(cube)
    mean_burnt_area = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, 
                                     weights=grid_areas).data.data
    q95 = cube.collapsed(['longitude', 'latitude'], iris.analysis.PERCENTILE, 
                         percent = [99]).data.data
    for tstep in range(cube.shape[0]): 
        cube.data.mask[tstep][cube.data[tstep] <= q95[tstep]] =  True
    grid_areas = iris.analysis.cartography.area_weights(cube)
    mean_extreme_burnt_area = cube.collapsed(['longitude', 'latitude'], 
                                             iris.analysis.MEAN, weights=grid_areas).data.data
    
    climatology, anomaly, ratio = \
        climtatology_difference(np.array([mean_burnt_area, mean_extreme_burnt_area]), nmnths)
    
    df = pd.DataFrame({
        'time': time_points,
        'mean_burnt_area': mean_burnt_area,
        'mean_burnt_area_climateology': np.tile(climatology[0], math.ceil(len(mean_burnt_area)/nmnths))[0:len(mean_burnt_area)],
        'mean_burnt_area_anomaly': anomaly[0],
        'mean_burnt_area_ratio': ratio[0],
        'p95_burnt_area': mean_extreme_burnt_area,
        'p95_burnt_area_climateology': np.tile(climatology[1], math.ceil(len(mean_burnt_area)/nmnths))[0:len(mean_burnt_area)],
        'p95_burnt_anomaly': anomaly[1],
        'p95_burnt_ratio': ratio[1]
    })
    
    df.to_csv(csv_dir_out + region + '/burnt_area_data' + filename_extra + '.csv', index=False)
    
    try:
        ax1 = axes[i]
    except:
        set_trace()
    ax2 = ax1.twinx()

    ax2.plot(time_points, mean_extreme_burnt_area, 'r--', label='Extreme (95%)')
    ax1.plot(time_points, mean_burnt_area, 'b-', label='Mean')
    ax2.plot(time_points, mean_extreme_burnt_area, 'r--')

    ax1.set_title(region)
    #ax1.set_ylabel('Mean', color='b')
    #ax2.set_ylabel('Extreme', color='r')

    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')




def for_summery(regions, mnths, sub_year = True, *args, **kw):
    # Set up a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))#, constrained_layout=True)
    axes = axes.flatten()  # Flatten to make it easier to loop through

    for i, region in enumerate(regions): 
        if sub_year: 
            for_region(region, i, fig, axes, mnths[i], 'for_event_months', *args, **kw)
        else:
            for_region(region, i, fig, axes, *args, **kw)
    # Tidy up the layout
    fig.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.08, wspace=0.3, hspace=0.3)
    #plt.tight_layout()
    plt.suptitle("Burnt Area Time Series by Region", fontsize=16, y=1.02)
    plt.subplots_adjust(top=0.88)  # Leave space for the main title
    # Shared axis labels (like mtext in R)
    fig.text(0.01, 0.5, 'Mean Burnt Area', va='center', ha='center',
             rotation='vertical', color='b', fontsize=12)
    fig.text(0.985, 0.5, 'Extreme Burnt Area (≥95%)', va='center', ha='center', 
             rotation='vertical', color='r', fontsize=12)

    # Optional: Shared title
    fig.suptitle("Burnt Area Time Series by Region", fontsize=16, y=1.03)

    fname = 'figs/regions_burnt_area_obsvered_ts'
    if sub_year:
        fname = fname + "for_event_months"
    fname = fname + '.png'
    plt.savefig(fname)

def run_both_mean_and_event(regions, mnths, *args, **kw):
    for_summery(regions, mnths, *args, **kw)
    for_summery(regions, mnths, False, *args, **kw)

if __name__=="__main__":
    # List of regions
    regions = ["Amazon", "Congo", "LA", "Pantanal", "NEIndia", "Alberta"]
    csv_dir_out = 'data/data/driving_data2425/'
    mnths = [[0, 1, 2], [5, 6, 7], [0], [5, 6, 7], [3], [6]]

    run_both_mean_and_event(regions, mnths, csv_dir_out = csv_dir_out)

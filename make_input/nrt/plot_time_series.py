import iris
import iris.analysis
import iris.analysis.cartography
import matplotlib.pyplot as plt
import glob
import iris.plot as iplt
from pdb import set_trace

import numpy.ma as ma
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import iris.coord_categorisation as icc

import matplotlib.dates as mdates
import pandas as pd
import numpy as np

dir = "data/data/driving_data2526/<<REGION>>/nrt/factual/"
def plot_time_series_for_region(region, dir):
    region = region.replace(' ', '_')
    dir = dir.replace("<<REGION>>", region)
    files = glob.glob(dir + "*.nc")
    
    time_series_cubes = []
    
    for f in files:
        cube = iris.load_cube(f)
        
        # Ensure lat/lon bounds exist (needed for area weights)
        if not cube.coord('latitude').has_bounds():
            cube.coord('latitude').guess_bounds()
        if not cube.coord('longitude').has_bounds():
            cube.coord('longitude').guess_bounds()
    
        # Compute grid cell areas
        weights = iris.analysis.cartography.area_weights(cube)
    
        # Collapse over lat/lon using weights
        print(f)
        
        cube.data = ma.masked_invalid(cube.data)
        ts = cube.collapsed(
            ['latitude', 'longitude'],
            iris.analysis.MEAN,
            weights=weights
        )
        #set_trace()
        time_series_cubes.append((f.split('/')[-1].split('.nc')[0], ts))
    
    n = len(time_series_cubes)  # or however many plots you want
    
    fig, axes = plt.subplots(nrows=n, ncols=2, sharex=True, figsize=(8, 2*n))
    
    # If n == 1, axes isn't a list → fix that
    if n == 1:
        axes = [axes]
    
    axes = axes.T.flatten()
    
    for ax, (name, ts) in zip(axes[0:len(time_series_cubes)], time_series_cubes):
        time = ts.coord('time').units.num2date(ts.coord('time').points)
        time = pd.to_datetime([str(t) for t in time])
    
        ax.plot(time, ts.data)
        ax.set_title(name)
        ax.set_ylabel(str(ts.units))
        ax.tick_params(axis='x', rotation=45)
    
    
    def cyclic_mean(arr, n):
        return np.array([arr[i::n].mean() for i in range(n)])
    
    
    for ax, (name, ts) in zip(axes[len(time_series_cubes):(2*len(time_series_cubes))], 
                              time_series_cubes):
        time = ts.coord('time').units.num2date(ts.coord('time').points)
        time = pd.to_datetime([str(t) for t in time])
    
        y = np.flip(ts.data)
        try:
            icc.add_year(ts, 'time')
        except:
            pass
        if len(np.unique(ts.coord('year').points) ) == len(ts.coord('year').points):
            lyr = 1
        else:
            lyr = 12
        tail = y[0:lyr]
        clim = cyclic_mean(y, lyr)
        y -= clim[np.arange(len(y)) % lyr]
        y[0:lyr] = np.nan
        y = np.flip(y)
        ax.plot(time, y)
        ax.set_title(name)
        ax.set_ylabel(str(ts.units))
        ax.tick_params(axis='x', rotation=45)

    # Format x-axis once (on bottom plot)
    #axes[-1].tick_params(axis='x', rotation=45)
    
    fig.autofmt_xdate()
    plt.tight_layout()
    
    out_file = dir.split(region)
    out_file = out_file[0] + '_'.join(out_file[1].split('/')) + '_' \
                     + "time_series" + '_' + region + '.png'
    plt.savefig(out_file, dpi = 300) 

if __name__=="__main__":
    regions = ["Midwestern Canadian Shield forests", 
               "Chilean Temperate Forests and Matorral", 
               "Southeast South Korea", 
               "Northwest Iberia", 
               "Scottish Highlands"
               ]
    [plot_time_series_for_region(region, dir) for region in regions]

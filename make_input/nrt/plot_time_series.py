import iris
import iris.analysis
import iris.analysis.cartography
import matplotlib.pyplot as plt
import glob
import iris.plot as iplt
from pdb import set_trace


dir = "data/data/driving_data2526/Midwestern_Canadian_Shield_forests/nrt/factual/"
files = glob.glob(dir + "*.nc")

time_series_cubes = []

for f in files:
    try:
        cube = iris.load_cube(f)
    except:
        set_trace()
    
    # Ensure lat/lon bounds exist (needed for area weights)
    if not cube.coord('latitude').has_bounds():
        cube.coord('latitude').guess_bounds()
    if not cube.coord('longitude').has_bounds():
        cube.coord('longitude').guess_bounds()

    # Compute grid cell areas
    weights = iris.analysis.cartography.area_weights(cube)

    # Collapse over lat/lon using weights
    ts = cube.collapsed(
        ['latitude', 'longitude'],
        iris.analysis.MEAN,
        weights=weights
    )
    #set_trace()
    time_series_cubes.append((f.split('/')[-1].split('.nc')[0], ts))

import matplotlib.dates as mdates
import pandas as pd
import numpy as np
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

import iris.coord_categorisation as icc


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
    y -= tail[np.arange(len(y)) % lyr]
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
plt.savefig(dir + 'time_series.png', dpi = 300) 
set_trace()

for name, ts in time_series_cubes:
    plt.figure()
    #iplt.plot(ts)
    #set_trace()
    time = ts.coord('time').units.num2date(ts.coord('time').points)

    # Convert to pandas datetime (this forces compatibility)
    time = pd.to_datetime([str(t) for t in time])

    plt.plot(time, ts.data)
    plt.gcf().autofmt_xdate()
    
    plt.xticks(rotation=45)
    
    plt.title(name)
    #plt.xlabel('Time')
    plt.ylabel(str(ts.units))
    plt.show()


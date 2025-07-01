import glob
import os
import sys
sys.path.append('libs/')
from plot_maps import *
from  constrain_cubes_standard import *
from state_of_wildfires_region_info  import get_region_info
import matplotlib.pyplot as plt
import iris
import numpy as np


dir1 = 'outputs/outputs_scratch/ConFLAME_nrt-attribution9/'
dir2 = '-2425/samples/_19-frac_points_0.5/'
def map_attribution_for_region(dir1, dir2, region, variable = 'Evaluate', nfiles = 1000):   
    region_info = get_region_info(region)[region]
    temp_file = 'temp/attribution_where-' + region_info['dir'] + '-' + \
                variable + '-' + str(nfiles) + '.nc' 
    if os.path.exists(temp_file):
        count_map = iris.load_cube(temp_file)
    else:
        # Which month index?
        month_idx = region_info['mnths']    
        year = region_info['years'][0]  
        
        fact_dir = dir1 + region_info['dir'] + dir2  + '/'
        cfact_dir = fact_dir + 'counterfactual-/' + variable + '/'
        fact_dir = fact_dir + 'factual-/' + variable + '/'
        
        fact_files = sorted(glob.glob(os.path.join(fact_dir, 'sample-pred*.nc')))[0:nfiles]
        cfact_files = sorted(glob.glob(os.path.join(cfact_dir, 'sample-pred*.nc')))[0:nfiles]
    
        # Load first cube to get grid info
        count_map = iris.load_cube(fact_files[0])[0]
        count_map.data[:] = 0.0
        
        def load_file_month(file):
            cube = iris.load_cube(file)
            cube0 = cube.copy()
            cube = sub_year_range(cube, [year, year])
            cube = sub_year_months(cube, month_idx)
            try:
                cube = cube.collapsed('time', iris.analysis.MEAN)    
            except:
                pass
            return cube.data
        
        
        for f_file, c_file in zip(fact_files, cfact_files):
            fact_data = load_file_month(f_file)
            cfact_data = load_file_month(c_file)
            
            # Compare and count
            count_map.data += (fact_data > cfact_data)
        
        count_map.data = count_map.data * 100.0/len(fact_files)
        #count_map.data[count_map.data<50.0] = 0.0
        iris.save(  count_map,   temp_file)
    plt.figure(figsize=(10, 6))
    
    plot_map_sow(count_map, get_region_info(region)[region]['shortname'], 
                        cmap=SoW_cmap['gradient_hues'], 
                        #levels=[0, 10, 20, 30, 40, 50, 60, 70, 80 ,90, 100],
                        levels=[50, 70, 80, 85, 90, 95, 100],
                        extend = 'min', cbar_label = "P(Factual > Counterfactual))")
#plt.colorbar(label='Count of Factual > Counterfactual')

#   plt.title(f'Count map: factual > counterfactual (month index {month_idx})')
    fig_name = 'figs/attribution_map-' + region + '-' + variable + '-' + str(nfiles) + '.png'
    plt.savefig(fig_name, dpi = 300)

regions = ['Amazon', 'Pantanal', 'Congo', 'LA']
for region in regions:
    map_attribution_for_region(dir1, dir2, region)    

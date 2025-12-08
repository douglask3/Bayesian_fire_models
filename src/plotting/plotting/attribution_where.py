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
import calendar

def month_range(month_strings):
    """
    Convert a list like ['01', '02', '03'] → 'Jan-Mar'.
    Assumes the months are consecutive and cover one calendar span.
    """
    # Convert strings → ints and sort (just in case)
    months = sorted(int(m) for m in month_strings)

    # Get three‑letter abbreviations (calendar.month_abbr[0] is '')
    start = calendar.month_abbr[months[0]]
    end   = calendar.month_abbr[months[-1]]

    # Same start & end → single month (e.g. ['07'] → 'Jul')
    return start if start == end else f"{start}-{end}"

dir1 = 'outputs/outputs_scratch/ConFLAME_nrt-attribution9/'
dir2 = '-2425/samples/_19-frac_points_0.5/'
def map_attribution_for_region(dir1, dir2, region, ax = None, variable = 'Evaluate', nfiles = 1000):   
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
        
        count_map.data = count_map.data * 100.0/(len(fact_files)-1)
        #count_map.data[count_map.data<50.0] = 0.0
        iris.save(  count_map,   temp_file)
    if ax is None: plt.figure(figsize=(10*0.7, 6*0.7))
    
    title = get_region_info(region)[region]['shortname'] + ' (' \
                + month_range(get_region_info(region)[region]['mnths']) + ')'
    add_cbar = ax is None
    
    return plot_map_sow(count_map, title,
                        cmap=SoW_cmap['gradient_hues'], 
                        #levels=[0, 10, 20, 30, 40, 50, 60, 70, 80 ,90, 100],
                        #levels=[50, 70, 80, 85, 90, 95, 100],
                        levels = [0, 33, 50, 66, 80, 85, 90, 99, 100],
                        extend = 'neither', cbar_label = "P(Factual > Counterfactual))",
                        add_cbar = add_cbar,
                        ax = ax)
#plt.colorbar(label='Count of Factual > Counterfactual')

#   plt.title(f'Count map: factual > counterfactual (month index {month_idx})')
    if ax is None:
        fig_name = 'figs/attribution_map-' + region + '-' + variable + '-' + \
                    str(nfiles) + '.png'
        plt.savefig(fig_name, dpi = 300)

from matplotlib.gridspec import GridSpec
widths = [1.5, 1]
heights = [2, 3]
fig = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 2, width_ratios=widths, height_ratios=heights, figure=fig)

regions = ['Amazon', 'LA', 'Congo', 'Pantanal']

#axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(len(regions))]
axes = [
    fig.add_subplot(gs[i // 2, i % 2], projection=ccrs.PlateCarree())
    for i in range(len(regions))
]

for ax, region in zip(axes, regions):
    img = map_attribution_for_region(dir1, dir2, region, ax=ax)  
    #ax.axis('off')
    if region == 'Congo':
        #ax.axis('off')
        levels = [0, 33, 50, 66, 80, 85, 90, 99, 100]
        cbar_label = "P(Factual > Counterfactual)"
        cbar = plt.colorbar(img, ax=ax, ticks=levels, orientation='horizontal')
        cbar.set_label(cbar_label, labelpad=10, loc='center')
        cbar.ax.xaxis.set_label_position('bottom')


        # Add top labels using a twin axis
        cax = cbar.ax
        cax_top = cax.secondary_xaxis('top')

        # Label midpoints of each range
        top_tick_pos = [100/16, 25, 100*4.5/8, 100*6.5/8, 100*7.5/8]  # midpoints of your bins
        top_tick_labels = [
            "Unlikely",
            "About as\nlikely as not",
            "Likely",
            "Very\nlikely",
            "Virtually\ncertain"
        ]
        range_edges = [0, 100/8, 100*3/8, 100*6/8, 100*7/8, 100]
        cax_top.set_ticks(range_edges, minor=False)
        cax_top.set_xticklabels([''] * len(range_edges))  # No labels on edge ticks
        for pos, label in zip(top_tick_pos, top_tick_labels):
            cax.text(pos, 1.4, label, ha='center', va='bottom', 
                         fontsize=9, rotation=0, transform=cax_top.transData)
        
        cax_top.tick_params(axis='x', length=10, width = 1.5, direction='out', top=True)
        #cax_top.set_ticks(top_tick_pos)
        #cax_top.set_xticklabels(top_tick_labels)
        #cax_top.tick_params(axis='x')#, rotation=30)
#plt.tight_layout()
plt.savefig("figs/attrbution_where.png", dpi = 300) 
plt.savefig("figs/attrbution_where.pdf")
plt.show() 

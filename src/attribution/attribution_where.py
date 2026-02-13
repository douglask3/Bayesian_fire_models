import glob
import os
import sys
sys.path.append('libs/')
sys.path.append('SoW_info/')
from plot_maps import *
from  constrain_cubes_standard import *
from state_of_wildfires_region_info  import get_region_info
import matplotlib.pyplot as plt
import iris
import numpy as np
import calendar
from matplotlib.gridspec import GridSpec
from  pathlib import Path

def month_range(month_strings):
    """
    Convert a list like ['01', '02', '03'] → 'Jan-Mar'.
    Assumes the months are consecutive and cover one calendar span.
    """
    # Convert strings → ints and sort (just in case)
    try:
        months = sorted(int(m) for m in month_strings)
    except:
        set_trace()
    # Get three‑letter abbreviations (calendar.month_abbr[0] is '')
    start = calendar.month_abbr[months[0]]
    end   = calendar.month_abbr[months[-1]]

    # Same start & end → single month (e.g. ['07'] → 'Jul')
    return start if start == end else f"{start}-{end}"

def map_attribution_for_region(dir1, dir2, region, 
                              ax = None, variable = 'Evaluate', nfiles = 1000,
                              temp_filename = '',
                              add_cbar = None, month_idx = None, year = None,
                              title = ''):   
    
    
    
    temp_file = dir1 + 'data_store/attribute_where/' + dir2.replace('samples', '') + \
                '-' + temp_filename + variable + '-' + str(nfiles) + '.nc'  
    if os.path.exists(temp_file):
        count_map = iris.load_cube(temp_file)
    else:
        Path(temp_file).parent.mkdir(parents=True, exist_ok=True)
        # Which month index?            
        fact_dir = dir1 + region + dir2  + '/'
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
            if year is not None: cube = sub_year_range(cube, [year, year])
            if month_idx is not None: cube = sub_year_months(cube, month_idx)
            try:
                cube = cube.collapsed('time', iris.analysis.MEAN)    
            except:
                pass
            try:
                return cube.data
            except:
                set_trace()
        
        
        for f_file, c_file in zip(fact_files, cfact_files):
            fact_data = load_file_month(f_file)
            cfact_data = load_file_month(c_file)
            
            # Compare and count
            count_map.data += (fact_data > cfact_data)
        
        count_map.data = count_map.data * 100.0/(len(fact_files)-1)
        #count_map.data[count_map.data<50.0] = 0.0
        iris.save(  count_map,   temp_file)
    if ax is None: plt.figure(figsize=(10*0.7, 6*0.7))
    
    if month_idx is not None:
        if title is not None:
            title += '(' + month_range(month_idx) + ')'
        else: 
            title = month_range(month_idx)

    if add_cbar is None:
        add_cbar = ax is None
    
    return plot_map_sow(count_map, title,
                        cmap=SoW_cmap['gradient_hues'], 
                        levels = [0, 33, 50, 66, 80, 85, 90, 99, 100],
                        extend = 'neither', cbar_label = "P(Factual > Counterfactual)",
                        add_cbar = add_cbar,
                        ax = ax)
#plt.colorbar(label='Count of Factual > Counterfactual')

#   plt.title(f'Count map: factual > counterfactual (month index {month_idx})')
    if ax is None:
        fig_name = 'figs/attribution_map-' + region + '-' + variable + '-' + \
                    str(nfiles) + '.png'
        plt.savefig(fig_name, dpi = 300)


def add_attribubtion_map_cbar(img, ax, levels = [0, 33, 50, 66, 80, 85, 90, 99, 100], 
                              top_tick_pos = [100/16, 25, 100*4.5/8, 100*6.5/8, 100*7.5/8],
                              range_edges = [0, 100/8, 100*3/8, 100*6/8, 100*7/8, 100],
                              top_tick_labels = ["Unlikely", "About as\nlikely as not", 
                                                 "Likely", "Very\nlikely", 
                                                 "Virtually\ncertain"],
                              cbar_label = "P(Factual > Counterfactual)"):

    cbar = plt.colorbar(img, ax=ax, ticks=levels, orientation='horizontal')
    cbar.set_label(cbar_label, labelpad=10, loc='center')
    cbar.ax.xaxis.set_label_position('bottom')
    
    cax = cbar.ax
    cax_top = cax.secondary_xaxis('top')
    cax_top.set_ticks(range_edges, minor=False)
    cax_top.set_xticklabels([''] * len(range_edges))  # No labels on edge ticks
    for pos, label in zip(top_tick_pos, top_tick_labels):
        cax.text(pos, 1.4, label, ha='center', va='bottom', 
                 fontsize=9, rotation=0, transform=cax_top.transData)
            
        cax_top.tick_params(axis='x', length=10, width = 1.5, direction='out', top=True)


def plot_confidence_in_attribution(dir1, dir2, region):
    eg_file  = list(Path(dir1 + region + dir2 + '/').rglob('*.nc'))[0]
    eg_cube = iris.load_cube(eg_file)
    
    nplots = eg_cube.shape[0] + 2
    nrows = int(np.ceil(np.sqrt(nplots)))
    ncols = int(np.ceil(nplots/nrows))
    
    fig, axes = set_up_sow_plot_windows(nrows, ncols, eg_cube[0],  size_scale = 3)

    for mnth, ax in zip(eg_cube.coord('month_number').points, axes[:-1]):
        
        month_idx = '0' + str(mnth) if mnth <9 else str(mnth)#
        map_attribution_for_region(dir1, dir2, region, 
                                       temp_filename = '-base-' + month_idx + '-', 
                                       ax = ax, add_cbar = False, month_idx = [month_idx])
    
    img = map_attribution_for_region(dir1, dir2, region, 
                                    temp_filename = '-base-',
                                    ax = axes[nplots-2], add_cbar = False) 
    if (nplots-1) < (ncols * nrows):    
        for ax in axes[(nplots-1):]: ax.set_visible(False)


    pos1 = axes[-nrows].get_position()
    pos2 = axes[-1].get_position()
    x0 = pos1.x0
    x1 = pos2.x1
    y1 = pos1.y1  # top of the top row
    height = 0.8  # thickness of the colorbar

    cbar_ax = fig.add_axes([x0, y1 - height*0.1, x1 - x0, height]) 
    cbar_ax.set_visible(False)
    add_attribubtion_map_cbar(img, cbar_ax)
    
    plt.savefig("figs/attrbution_where-base" + region + ".png", dpi = 300) 


if __name__=="__main__":
    dir1 = 'outputs/outputs_scratch/ConFLAME_nrt-attribution9/'
    dir2 = '-2425/samples/_19-frac_points_0.5/'
    widths = [1.5, 1]
    heights = [2, 3]
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, width_ratios=widths, height_ratios=heights, figure=fig)
    
    regions = ['Amazon', 'LA', 'Congo', 'Pantanal']
    
    
    axes = [
        fig.add_subplot(gs[i // 2, i % 2], projection=ccrs.PlateCarree())
        for i in range(len(regions))
    ]

    for ax, region in zip(axes, regions):
        img = map_attribution_for_region(dir1, dir2, region, ax=ax)  
        #ax.axis('off')
        if region == 'Congo':
            
            add_attribubtion_map_cbar(img, ax)

    plt.savefig("figs/attrbution_where.png", dpi = 300) 
    plt.savefig("figs/attrbution_where.pdf")
    plt.show() 

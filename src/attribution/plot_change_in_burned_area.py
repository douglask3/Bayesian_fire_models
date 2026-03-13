import sys
sys.path.append('.')
sys.path.append('plotting/')
sys.path.append('src/')
sys.path.append('src/attribution/')
sys.path.append('SoW_info/')
from  pathlib import Path

from plot_BA_climateology import *
from plot_maps import *

import numpy as np

def ratio_levels(ratios, n_levels = 7, *args, **kw):
    assert n_levels % 2 == 1, "N must be odd"

    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]

    half = (n_levels - 1) // 2

    log_r = np.log10(ratios)
    max_log = np.max(np.abs(log_r))

    # target max ratio
    max_ratio = 10**max_log

    # "nice" numbers sequence
    base = np.array([1, 2, 5])
    powers = np.arange(-10, 10)

    try:
        nice = (base[:, None] * 10.0**powers).ravel()
    except:
        set_trace()
    nice = nice[nice > 1]
    nice.sort()

    nice = nice[nice <= max_ratio]

    if len(nice) < half:
        # extend if needed
        nice = nice[:half]
    else:
        idx = np.linspace(0, len(nice)-1, half).astype(int)
        nice = nice[idx]

    levels = np.concatenate([1/nice[::-1], [1], nice])
    return levels

def plot_change_in_burned_area(dir1, dir2, region, run_name = 'Evaluate', obs_file = None,
                               out_dir = "figs/", percentiles = [5, 95]):
    #if obs_file is not None:
    #    #filename = "data/data/driving_data_base/" + region +"/burnt_area.nc"
    #    anomaly, climatology = open_netcdf_and_find_clim(obs_file)

    dir = dir1 + region + dir2 + '/'
    
    # Returns a list of Path objects for directories only
    experiments = [d for d in Path(dir).iterdir() if d.is_dir()]
    
    
    factual = [dir for dir in experiments if dir.name[0:7] == 'factual']
    if len(factual) > 1:
        set_trace()
    if len(factual) == 0:
        print("ERROR: no factual experiment")
        return
    factual = factual[0]
    counterfactual = [dir for dir in experiments if 'counterfactual' in dir.name]

    nc_files = list((factual / run_name).rglob('*pred*.nc'))
    

    def amplifcation_from_file(file):

        f_cube = iris.load_cube(factual / run_name / file.name)
        f_a_cube = f_cube.collapsed('time', iris.analysis.MEAN)
        def for_cf(dir):
            
            out_dir = Path(str(dir).replace('counterfactual', 'cf_minus_f'))
            out_dir1 = out_dir / 'monthly'
            out_dir2 = out_dir / 'temporal_average'
            
            out_file1 = out_dir1 / file.name
            out_file2 = out_dir2 / file.name

            if out_file1.is_file() and out_file2.is_file():
                out1 = iris.load_cube(out_file1)
                out2 = iris.load_cube(out_file2)
            else:
                out_dir1.mkdir(parents=True, exist_ok=True)
                out_dir2.mkdir(parents=True, exist_ok=True)
            
                cf_cube = iris.load_cube(dir / run_name / file.name)
            
                out1 = f_cube / cf_cube
                out2 = f_a_cube / cf_cube.collapsed('time', iris.analysis.MEAN)
                iris.save(out1, out_file1)
                iris.save(out2, out_file2)
            return(out1, out2)
        
        out = [[f_cube, f_a_cube]]
        for dir  in counterfactual:
            out += [for_cf(dir)]
        #out =  [for_cf(dir) for dir in counterfactual]
        # + out
        return out

    out = [amplifcation_from_file(file) for file in nc_files]
    out = np.array(out)
    
    def merge_realization(i,j):
        return iris.cube.CubeList(out[:,i, j]).merge_cube()
    
    out_merge = [[merge_realization(i, j) for j in range(out.shape[2])]\
                     for i in range(out.shape[1])]
    out_merge = np.array(out_merge)
    
    summery = [[cube.collapsed('realization', iris.analysis.PERCENTILE, percent = percentiles) \
                for cube in cubes] for cubes in out_merge]
    #set_trace()
    summery = np.array(summery)
    #set_trace()
    nrows = out_merge[0][0].shape[1] + 1
    ncols = (out_merge.shape[0])*len(percentiles)
    eg_cube = out_merge[0][0]

    def find_levels(ii, n_levels = 7, FUN = auto_pretty_levels, *args, **kw):
        
        try:    
            all_fact = np.array([np.append(summery[i][0].data.flatten(), \
                             summery[i][1].data.flatten()) for i in ii])
        except:
            all_fact[np.abs(all_fact)<0.0001] = 0.0
        return FUN(all_fact,   n_levels=n_levels, ignore_v = 0.0, *args, **kw)
    
    dlevels = find_levels(range(1, len(summery)), n_levels =7, FUN = ratio_levels)#ratio = 1, force0 = True)
    #set_trace()
    dlevels = np.array([0.25, 0.3, 0.33, 0.4, 0.5, 0.67, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    levels = find_levels([0])
    levels = np.append(0, levels)
    
    fig, axes = set_up_sow_plot_windows(nrows, ncols, eg_cube[0],  size_scale = 3)

    mnths = eg_cube.coord('month_number').points
    axi = 0

    def plot_map_fun(cube, levels, cmap, extend, axi, *args, **kw):
        return plot_map_sow(cube, '', levels = levels, add_cbar = False, 
                            cmap=SoW_cmap[cmap], extend = extend, ax = axes[axi], 
                            *args, **kw)
    
    def select_month(cube, mnthi):
        if len(cube.shape) == 3:
            out = [cube[mnthi]]
        else:
            out = cube[:, mnthi]
        return out

    npc = len(percentiles)
    for mnthi, mnth in zip(range(len(mnths)), mnths):
        month_idx = calendar.month_name[mnth]
        fact = select_month(summery[0][0], mnthi)
        for fi in range(npc):
            plot_map_fun(fact[fi], levels, 'gradient_red', 'max', axi)
                
            if fi == 0:
                axes[axi].text(-0.15, 0.5, month_idx,  transform=axes[axi].transAxes, 
                        va='center', ha='right', rotation=90)
            axi += 1
        for cfn in range(1, summery.shape[0]):
            cf = fact = select_month(summery[cfn][0], mnthi)
            for fi in range(npc):                
                plot_map_fun(cf[fi], dlevels, 'diverging_BlueRed', 'both', axi)
                axi += 1
    
    for fi in range(npc):
        if len(summery[0][1].shape) == 2:
            cube = summery[0][1]
        else:
            cube = summery[0][1][fi]
        
        img = plot_map_fun(cube, levels, 'gradient_red', 'max', axi)
        if fi == 0:
            axes[axi].text(-0.15, 0.5, 'Average',  transform=axes[axi].transAxes, 
                    va='center', ha='right', rotation=90)
        axi += 1
    #set_trace()
    def add_cbar(axi1, axi2, img):
        pos0 = axes[axi1].get_position()
        pos1 = axes[axi2].get_position()
        cbar_width = pos1.x1 - pos0.x0
        cax = fig.add_axes([pos0.x0, pos0.y0 - 0.05, cbar_width, 0.02])
        cbar = fig.colorbar(img, cax=cax, orientation='horizontal')

        if hasattr(img, "levels"):  # contourf case
            cbar.set_ticks(img.levels)
        
        # Get tick labels
        labels = cbar.ax.get_xticklabels()
        
        for i, label in enumerate(labels):
            label.set_rotation(45)
            
            if i % 2 == 0:
                # Bottom labels
                label.set_verticalalignment('top')
                label.set_y(-0.02)   # small downward shift
            else:
                # Top labels
                label.set_verticalalignment('bottom')
                label.set_y(1.1)    # move above the bar
                #label.set_horizontalalignment('right')

        
        '''
        # 1. Create a twin axis for the colorbar
        cbar_ax = cbar.ax
        
        # 2. Sync ticks and limits
        # 2. Create a twin x-axis for the top of the colorbar
        twin_ax = cbar_ax.twiny()
        
        # 3. Synchronize the twin axis with the colorbar's scale
        ticks = cbar.get_ticks()
        set_trace()
        twin_ax.set_xlim(cbar_ax.get_xlim())
        twin_ax.set_xticks(ticks)
        twin_ax.set_xticklabels([f'{t:.1f}' for t in ticks])
        #set_trace() 
        # 4. Toggle visibility to alternate labels
        for i, (l_bot, l_top) in enumerate(zip(cbar_ax.get_xticklabels(), twin_ax.get_xticklabels())):
            if i % 2 == 0:
                l_top.set_visible(False) # Even index: label stays at bottom
            else:
                l_bot.set_visible(False) # Odd index: label moves to top
        
        #plt.show()        
        #set_trace()
        cbar.ax.xaxis.set_ticks_position('both')

        # 2. Loop through ticks to alternate label visibility
        for i, tick in enumerate(cbar.ax.xaxis.get_major_ticks()):
            if i % 2 == 0:
                tick.label1.set_visible(True)   # Bottom label ON
                tick.label2.set_visible(False)  # Top label OFF
            else:
                tick.label1.set_visible(False)  # Bottom label OFF
                tick.label2.set_visible(True)   # Top label ON
        '''
    add_cbar(axi-npc, axi-1, img)
    for cfn in range(1, summery.shape[0]):
        for fi in range(npc):    
            if len(summery[cfn][1].shape) == 2:
                cube = summery[cfn][1]
            else:
                cube = summery[cfn][1][fi]
            img = plot_map_fun(cube, dlevels, 'diverging_BlueRed', 'both', axi)
            axi += 1
    add_cbar(axi-npc*2, axi-1, img)

    titles = [factual.name[:-1]] + [cf.name[:-1] for cf in counterfactual]

    for i in range(summery.shape[0]):
        for j, title in enumerate(percentiles):
            axes[i*2].set_title(str(title) + '%', fontsize=10)
        #axes[i*2 + 1].set_title('95%', fontsize=10)

        pos0 = axes[i*2].get_position()
        pos1 = axes[i*2 + len(percentiles) -1].get_position()

        # Calculate the center between the two columns
        center_x = (pos0.x0 + pos1.x1) / 2
        # Set the height slightly above the individual titles
        top_y = pos0.y1 + 0.01 

        fig.text(center_x, top_y, titles[i], 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    #set_trace()
    plt.savefig(out_dir + "attrbution_increase_map" + region + "-".join([str(pc) for pc in percentiles]) + ".png", dpi = 200) 


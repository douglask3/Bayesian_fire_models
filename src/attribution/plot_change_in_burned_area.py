import sys
sys.path.append('.')
sys.path.append('plotting/')
sys.path.append('src/')
sys.path.append('src/attribution/')
sys.path.append('SoW_info/')
from  pathlib import Path
from  attribution_where import *
from plot_BA_climateology import *
from plot_maps import *
from constrain_cubes_standard import *
import numpy as np

def ratio_levels(ratios, n_levels = 9, *args, **kw):
    assert n_levels % 2 == 1, "N must be odd"

    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]

    half = (n_levels - 1) // 2

    log_r = np.log10(ratios)
    max_log = np.percentile(np.abs(log_r), 100 * (1-1/(2*n_levels)))
    
    # target max ratio
    max_ratio = 10**max_log

    # "nice" numbers sequence
    base = np.array([1, 1.5, 2, 3, 4, 5])
    powers = np.arange(-10, 10)
    
    nice = (base[:, None] * 10.0**powers).ravel()
    nice = nice[nice > 1]
    nice.sort()

    nice = nice[nice <= max_ratio]
    #set_trace()
    if len(nice) < half:
        # extend if needed
        nice = nice[:half]
    else:
        idx = np.linspace(0, len(nice)-1, half).astype(int)
        nice = nice[idx]

    levels = np.concatenate([1/nice[::-1], [1], nice])
    return levels

def add_cbar(axi1, axi2, fig,  img, axes, height = None, ratio = False, cbar_label = ''):
    
    pos0 = axes[axi1].get_position()
    pos1 = axes[axi2].get_position()
    cbar_width = pos1.x1 - pos0.x0
    if height is None:
        y0 = pos0.y0 - 0.05
        height = 0.02
    else:
        y0 = pos0.y1 - height*0.5
        height = height * 0.5
        
    cax = fig.add_axes([pos0.x0, y0, cbar_width, height])
    cbar = fig.colorbar(img, cax=cax, orientation='horizontal')
    
    if hasattr(img, "levels"):  # contourf case
        cbar.set_ticks(img.levels)
    cbar.set_label(cbar_label, size=15, labelpad=10)
    # Get tick labels
    ticks = cbar.get_ticks()

    if ratio:
        labels = []
        for num in ticks:
            if num < 1:
                num = 1/num
                if int(num) == num:
                    num = int(num)
                else:
                    num = round(num, 2)
                labels.append('1/' +str(num))
            else:
                if int(num) == num:
                    num = int(num)
                else:
                    num = round(num, 2)
                labels.append(str(num))
    else:
        labels = [str(t) for t in ticks]

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)

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
            label.set_y(1.4)    # move above the bar
            #label.set_horizontalalignment('right')

    

def find_levels(summery, ii, n_levels = 7, FUN = auto_pretty_levels, fullYrOnly = False, *args, **kw):        
    try:   
        if fullYrOnly:
            all_fact = np.array([summery[i][1].data.flatten() for i in ii])
        else:
            all_fact = np.array([np.append(summery[i][0].data.flatten(), \
                                 summery[i][1].data.flatten()) for i in ii])
    except:
        all_fact[np.abs(all_fact)<0.0001] = 0.0
    try:
        out = FUN(all_fact, n_levels=n_levels, ignore_v = 0.0, *args, **kw)
    except:
        set_trace()
    return out


def plot_change_in_burned_area(dir1, dir2, region, run_name = 'Evaluate', obs_file = None,
                               out_dir = "figs/", percentiles = [5, 95], year = None, 
                               mnths = None, 
                               counterfactual_name = 'counterfactual', plot_att_only = False,
                               fullYrOnly = False, axes = None, out_figname_extra = ''):


    def open_cube(filename):
        cube = iris.load_cube(filename)
        cube0 = cube.copy()
        cube = sub_year_range(cube, [year])
        cube = sub_year_months(cube, mnths)
        
        if len(mnths) > 1: 
            try:
                cube = cube.collapsed('time', iris.analysis.MEAN)
            except:
                set_trace()
        return cube
    obs = open_cube(obs_file)

    dir = dir1 + region + dir2 + '/'
    
    experiments = [d for d in Path(dir).iterdir() if d.is_dir()]
    
    factuals = [dir for dir in experiments if dir.name[0:7] == 'factual']
    counterfactuals = [dir for dir in experiments if counterfactual_name in dir.name]

    

    def plot_for_f_cf(factual, counterfactual, obs):
        nc_files = list((factual / run_name).rglob('*pred*.nc'))
        #set_trace()
        def amplifcation_from_file(file, obs):
            
            out_dir = factual / ('month' + '_'.join([str(mnth) for mnth in mnths]))
                
            out_file = out_dir / file.name
            print(out_file)
            if out_file.is_file():
                f_cube = iris.load_cube(out_file)
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                f_cube = open_cube(factual / run_name / file.name)
                iris.save(f_cube, out_file)
            obs = obs  

            out_dir = Path(str(counterfactual).replace('/month', '/prob'))
            out_file = out_dir / file.name
            if out_file.is_file():
                prob =  iris.load_cube(out_file)
            else:
                prob = f_cube.copy()
                prob.data = np.exp(obs.data*np.log(f_cube.data) + \
                                   (1.0-obs.data)*np.log((1-f_cube.data)))
                
                out_dir.mkdir(parents=True, exist_ok=True)
                iris.save(prob, out_file)
            #set_trace()
            out_dir = Path(str(counterfactual).replace('counterfactual', 'cf_minus_f'))
            out_dir = out_dir / ('month' + '_'.join([str(mnth) for mnth in mnths]))
                
            out_file = out_dir / file.name
            print(out_file)
            if out_file.is_file():
                cf_cube = iris.load_cube(out_file)
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                
                cf_cube = open_cube(counterfactual / run_name / file.name)
                
                cf_cube = f_cube / cf_cube
                iris.save(cf_cube, out_file)
             
            direction = cf_cube.copy()
            direction.data = direction.data > 1 
            direction.data = direction.data.astype('float32')  

            out = [prob, f_cube, cf_cube, direction]
            return out

        def merge_realization(i):
            return iris.cube.CubeList(out[:,i]).merge_cube()

        def weighted_mean(i):
            out = out_merge[i][0].copy()
            out.data = np.quantile(a=out_merge[i].data, q=0.5, axis=0,
                                   weights=out_merge[0].data, method="inverted_cdf")
            return out
        
        def save_cubes(cube, fname):
            iris.save(cube, 'outputs/' + fname + '-2.nc')
        def load_cubes(fname):
            return iris.load_cube( 'outputs/' + fname + '-2.nc')

        try:
            obs_c = load_cubes(region + '-fire_mask')
            pval  = load_cubes(region + '-Likelihood')
            waf   = load_cubes(region + '-AF')
        except:
            
            set_trace()
            out = np.array([amplifcation_from_file(file, obs) for file in nc_files])
            out_merge = [merge_realization(i) for i in range(out.shape[1])]
        
        
            wfact = weighted_mean(1)
            waf = weighted_mean(2)
            pval = weighted_mean(3)*100
            waf.data[obs.data == 0] = out_merge[2].collapsed('realization', 
                                                             iris.analysis.PERCENTILE, 
                                                             percent =50).data[obs.data == 0]
        
            pval.data[obs.data == 0] = \
                    out_merge[3].collapsed('realization', 
                                           iris.analysis.MEAN).data[obs.data == 0]

            obs_c = obs.copy()
            ba95 = np.sort(obs.data.data[obs.data.data <100])#
            ba95 = ba95[np.where(ba95.cumsum()>(0.05*ba95.sum()))[0][0]]
            
            obs_c.data[:] = obs_c.data> ba95
            save_cubes(obs_c,  region + '-fire_mask')
            save_cubes(pval*100, region + '-Likelihood')
            save_cubes(waf,  region + '-AF')
        
        fig, axes = set_up_sow_plot_windows(1, 2, obs,  
                                            size_scale = 2 + obs.shape[1]/obs.shape[0],
                                            oma = [0.6, 0.1, 0.25, 0.25])
        
        #plot_map_sow(obs * 100, "Burned area (%)", add_cbar = True, extend = 'max',
        #             cmap=SoW_cmap["gradient_red"], use_pcolmesh = True, ax = axes[0],
        #             cbar_orientation = 'horizontal',
        #             cbar_lab_rotate = 45, cbar_top_and_bottom = True)

        levels = [0, 1/5, 1/2, 1/1.5, 1/1.1, 1, 1.1, 1.5, 2, 5, 9E9]
        tick_labels = ['0', '1/5', '1/2', '1/1.5', '1/1.1', 'no\nchange', 
                       '1.1', '1.5', '2', '5', '★']
        levels = [0, 1, 1.1, 1.5, 2, 5]
        tick_labels = ['0', 'no\nchange', 
                       '1.1', '1.5', '2', '5']
        
        plot_map_sow(waf, "Amplification factor", 
                     scatter_obs = obs_c,
                     add_cbar = True, extend = 'max',
                     levels = levels, tick_labels = tick_labels,
                     cmap=SoW_cmap["gradient_red"], use_pcolmesh = True, ax = axes[0],
                     cbar_orientation = 'horizontal', 
                     cbar_lab_rotate = 45, cbar_top_and_bottom = True)
        
        levels = [0, 1, 10, 33, 66, 90, 99, 100]
        range_edges = np.arange(8)*100/7
        top_tick_pos = range_edges[1:].copy() - 100/14

        top_tick_labels = ["Extremely\nUnlikely", "Very\nUnlikely", "Unlikely", 
                          "As likely\nas not","Likely", "Very\nLikely", 
                                                 "Virtually\nCertain"]
        
        img = plot_map_sow(pval, "Likelihood",scatter_obs = obs_c,
                        cmap=SoW_cmap['confidence_hues'], 
                        levels = levels,
                        extend = 'neither', cbar_label = "",
                        add_cbar = False, use_pcolmesh = True,
                        ax = axes[1], cbar_orientation = 'horizontal', 
                        cbar_lab_rotate = 45, cbar_top_and_bottom = True)

        add_attribubtion_map_cbar(img, axes[1], levels, 
                                  top_tick_labels= top_tick_labels, 
                                  range_edges = range_edges, top_tick_pos = top_tick_pos,
                                  cbar_label = '')
         
        #add_attribubtion_map_cbar(img, axes[2]) 
            
        out_name = 'figs/attribtuion_map' + region + 'summery-2.png' 
        #set_trace()
        #+ factual + counterfactual + '.png'
        plt.savefig(out_name, dpi = 300)
    for fact in factuals:
        for counter in counterfactuals:
            plot_for_f_cf(fact, counter, obs)

def plot_change_in_burned_area_all(dir1, dir2, region, run_name = 'Evaluate', obs_file = None,
                               out_dir = "figs/", percentiles = [5, 95], mnths = None, 
                               counterfactual_name = 'counterfactual', plot_att_only = False,
                               fullYrOnly = False, axes = None, out_figname_extra = ''):
    set_trace()
    if obs_file is not None:
        #filename = "data/data/driving_data_base/" + region +"/burnt_area.nc"
        anomaly, climatology = open_netcdf_and_find_clim(obs_file)
    set_trace()
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
    counterfactual = [dir for dir in experiments if counterfactual_name in dir.name]
    
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
                set_trace()
                out1 = f_cube / cf_cube
                out2 = f_a_cube / cf_cube.collapsed('time', iris.analysis.MEAN)
                iris.save(out1, out_file1)
                iris.save(out2, out_file2)
            return(out1, out2)
        
        out = [[f_cube, f_a_cube]]
        for dir  in counterfactual:
            out += [for_cf(dir)]
    
        return out

    out = [amplifcation_from_file(file) for file in nc_files]
    out = np.array(out)
    
    def merge_realization(i,j):
        return iris.cube.CubeList(out[:,i, j]).merge_cube()
    
    out_merge = [[merge_realization(i, j) for j in range(out.shape[2])]\
                     for i in range(out.shape[1])]
    out_merge = np.array(out_merge)
    set_trace()
    summery = [[cube.collapsed('realization', iris.analysis.PERCENTILE, percent = percentiles) \
                for cube in cubes] for cubes in out_merge]
    
    summery = np.array(summery)
    eg_cube = out_merge[0][0]
    if len(summery) == 2 and plot_att_only:
        nplots = nplots = eg_cube.shape[1] + 2
        nrows = int(np.ceil(np.sqrt(nplots)))
        ncols = int(np.ceil(nplots/nrows))
    else:
        nrows = eg_cube.shape[1] + 1
        ncols = (out_merge.shape[0])*len(percentiles)
    
    dlevels2 = find_levels(summery, range(0, len(summery)), 
                           n_levels = 11, FUN = ratio_levels, fullYrOnly = True)
    dlevels1 = find_levels(summery, range(0, len(summery)), 
                           n_levels = 11, FUN = ratio_levels, fullYrOnly = fullYrOnly)

    
    #dlevels = np.unique(np.sort(np.append(dlevels1, dlevels2)))
    #dlevels = np.array([0.25, 0.3, 0.33, 0.4, 0.5, 0.67, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    levels = find_levels(summery, [0], fullYrOnly = fullYrOnly)
    levels = np.append(0, levels)
    
    if axes is None:    
        fig, axes = set_up_sow_plot_windows(nrows, ncols, eg_cube[0],  size_scale = 3,
                                            oma = [1.0, 0.1, 0.25, 0.25])
        outplot = True
    else:
        outplot = False
    mnths = eg_cube.coord('month_number').points
    axi = 0

    def plot_map_fun(cube, levels, cmap, extend, axi, title = '', *args, **kw):
        return plot_map_sow(cube, title, levels = levels, add_cbar = False, 
                            cmap=SoW_cmap[cmap], extend = extend, ax = axes[axi], 
                            *args, **kw)
    
    def select_month(cube, mnthi):
        if len(cube.shape) == 3:
            out = [cube[mnthi]]
        else:
            out = cube[:, mnthi]
        return out

    npc = len(percentiles)
    
    if not fullYrOnly:
        for mnthi, mnth in zip(range(len(mnths)), mnths):
            month_idx = calendar.month_name[mnth]
            fact = select_month(summery[0][0], mnthi)
            
            if not plot_att_only:
                for fi in range(npc):
                    plot_map_fun(fact[fi], levels, 'gradient_red', 'max', axi)
                        
                    if fi == 0:
                        axes[axi].text(-0.15, 0.5, month_idx,  transform=axes[axi].transAxes, 
                                va='center', ha='right', rotation=90)
                    axi += 1
            
            for cfn in range(1, summery.shape[0]):
                cf = fact = select_month(summery[cfn][0], mnthi)
                for fi in range(npc):     
                    if len(summery) == 2 and plot_att_only and not fullYrOnly:
                        mnth_no = cf[0].coord('month_number').points
                        title = month_range([mnth_no])
                    else:
                        title = ''           
                    img1 = plot_map_fun(cf[fi], dlevels1, 'diverging_BlueRed', 'both', axi, 
                                 title = title)
                    axi += 1
            set_trace()
    if not plot_att_only:                
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
    
        add_cbar(axi-npc, axi-1, fig, img, axes)

    for cfn in range(1, summery.shape[0]):
        for fi in range(npc):    
            if len(summery[cfn][1].shape) == 2:
                cube = summery[cfn][1]
            else:
                cube = summery[cfn][1][fi]
            if len(summery) == 2 and plot_att_only:
                title = 'Mean over period'
            else:
                title = ''
            img2 = plot_map_fun(cube, dlevels2, 'diverging_BlueRed', 'both', axi, title = title)
            axi += 1
    if axi < len(axes):
        for ax in axes[axi:]: ax.set_visible(False)

    if len(summery) == 2 and plot_att_only:
        
        if axi > (len(axes) - ncols):
            add_cbar(axi+1, len(axes)-1, fig, img1, axes, 0.05, ratio = True, 
                     add_cbar = 'Monthly amplification')
            add_cbar(axi, axi, fig, img2, axes, 0.05, ratio = True, 
                     add_cbar = 'Mean amplification over period')
        else:
            add_cbar(axi, len(axes)-2, fig, img1, axes, 0.05, ratio = True)
            add_cbar(axi+2, axi+2, fig, img2, axes, 0.05, ratio = True)
    elif len(summery) == 1:
        
        add_cbar(axi-npc*2, axi-1, fig, img, axes, ratio = True)
    else:
        add_cbar(axi-npc*2, axi-1, fig, img2, axes, ratio = True)
   
    if not (len(summery) == 2 and plot_att_only):
        titles = [factual.name[:-1]] + [cf.name[:-1] for cf in counterfactual]
        for i in range(summery.shape[0]):
            for j, title in enumerate(percentiles):
                axes[i*npc].set_title(str(title) + '%', fontsize=10)
    
            pos0 = axes[i*npc].get_position()
            pos1 = axes[i*npc + len(percentiles) -1].get_position()
    
            # Calculate the center between the two columns
            center_x = (pos0.x0 + pos1.x1) / 2
            # Set the height slightly above the individual titles
            top_y = pos0.y1 + 0.01 
    
            fig.text(center_x, top_y, titles[i], 
                     ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    if outplot:
        plt.savefig(out_dir + out_figname_extra + "attrbution_increase_map" + region + \
                    "-".join([str(pc) for pc in percentiles]) + ".png", dpi = 400) 
    

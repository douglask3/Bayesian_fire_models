import sys
sys.path.append('.')
sys.path.append('plotting/')
sys.path.append('src/')
sys.path.append('src/attribution/')
sys.path.append('make_input/burnt_area/')
from plot_effect_rr import *
from attribution_where import *
from plot_BA_climateology import *
import imageio
from  pathlib import Path

def plot_all_attribution_scatter(dir1, dir2, regions, obs_dir, obs_file):
    if not isinstance(regions, list):
        regions = [regions]

    def plot_attribution_scatter_generic(counterfactual_name, 
                                         plot_name = "attribution_scatter",
                                         plot_FUN = plot_fact_vs_ratio):
        plot_attribution_scatter(regions, plot_name + counterfactual_name,
                                 dir1 = dir1, dir2 = dir2,
                                 obs_dir = obs_dir, obs_file = obs_file, 
                                 plot_FUN = plot_FUN)

    plot_attribution_scatter_generic("counterfactual")
    plot_attribution_scatter_generic("counterfactual-metmean")
    plot_attribution_scatter_generic("counterfactual", "rr_line",
                                     effect_ratio_and_rr_over_range)
    plot_attribution_scatter_generic("counterfactual-metmean", "rr_line",
                                     effect_ratio_and_rr_over_range)

def plot_confidence_in_attribution(dir1, dir2, region):
    eg_file  = list(Path(dir1 + get_region_info(region)[region]['dir'] + dir2 + '/').rglob('*.nc'))[0]
    eg_cube = iris.load_cube(eg_file)
    
    nplots = eg_cube.shape[0] + 2
    nrows = int(np.ceil(np.sqrt(nplots)))
    ncols = int(np.ceil(nplots/nrows))
    
    fig, axes = set_up_sow_plot_windows(nrows, ncols, eg_cube[0],  size_scale = 3)

    for mnth, ax in zip(eg_cube.coord('month_number').points, axes[:-1]):
        month_idx = '0' + str(mnth + 1) if mnth <9 else str(mnth + 1)#
        map_attribution_for_region(dir1, dir2, region, 
                                       temp_filename = '-base-' + month_idx + '-', 
                                       addRegion2Title = False,
                                       ax = ax, add_cbar = False, month_idx = [month_idx])

    img = map_attribution_for_region(dir1, dir2, region, 
                                    temp_filename = '-base-', addRegion2Title = False,
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

def plot_change_in_burned_area(dir1, dir2, region, run_name = 'Evaluate', obs_dir = None):
    if obs_dir is not None:
        filename = "data/data/driving_data_base/" + region +"/burnt_area.nc"
        anomaly, climatology = open_netcdf_and_find_clim(filename)

    dir = dir1 + get_region_info(region)[region]['dir'] + dir2 + '/'
    
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
            
                out1 = f_cube - cf_cube
                out2 = f_a_cube - cf_cube.collapsed('time', iris.analysis.MEAN)
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
        return iris.cube.CubeList(out[:,i, j]).merge_cube()*100
    
    out_merge = [[merge_realization(i, j) for j in range(out.shape[2])]\
                     for i in range(out.shape[1])]
    out_merge = np.array(out_merge)
    summery = [[cube.collapsed('realization', iris.analysis.PERCENTILE, percent = [5, 95]) \
                for cube in cubes] for cubes in out_merge]
    summery = np.array(summery)
    
    nrows = out_merge[0][0].shape[1] + 1
    ncols = (out_merge.shape[0])*2
    eg_cube = out_merge[0][0]

    def find_levels(ii, n_levels = 7, *args, **kw):
        
        all_fact = np.array([np.append(summery[i][0].data.flatten(), \
                             summery[i][1].data.flatten()) for i in ii])
        all_fact[np.abs(all_fact)<0.0001] = 0.0
        return auto_pretty_levels(all_fact,   n_levels=7, ignore_v = 0.0)
    
    dlevels = find_levels(range(1, 3))
    levels = find_levels([0])
    levels = np.append(0, levels)
    
    fig, axes = set_up_sow_plot_windows(nrows, ncols, eg_cube[0],  size_scale = 3)

    mnths = eg_cube.coord('month_number').points
    axi = 0

    def plot_map_fun(cube, levels, cmap, extend, axi, *args, **kw):
        return plot_map_sow(cube, '', levels = levels, add_cbar = False, 
                            cmap=SoW_cmap[cmap], extend = extend, ax = axes[axi], 
                            *args, **kw)
    

    for mnthi, mnth in zip(range(len(mnths)), mnths):
        month_idx = calendar.month_name[mnth]
        fact = summery[0][0][:, mnthi]
        for fi in range(2):
            plot_map_fun(fact[fi], levels, 'gradient_red', 'max', axi)
                
            if fi == 0:
                axes[axi].text(-0.15, 0.5, month_idx,  transform=axes[axi].transAxes, 
                        va='center', ha='right', rotation=90)
            axi += 1
        for cfn in range(1, summery.shape[0]):
            cf = summery[cfn][0][:, mnthi]
            for fi in range(2):                
                plot_map_fun(cf[fi], dlevels, 'diverging_BlueRed', 'both', axi)
                axi += 1

    for fi in range(2):
        img = plot_map_fun(summery[0][1][fi], levels, 'gradient_red', 'max', axi)
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
        fig.colorbar(img, cax=cax, orientation='horizontal')

    add_cbar(axi-2, axi-1, img)
    for cfn in range(1, summery.shape[0]):
        for fi in range(2):
            img = plot_map_fun(summery[cfn][1][fi], dlevels, 'diverging_BlueRed', 'both', axi)
            axi += 1
    add_cbar(axi-4, axi-1, img)

    titles = [factual.name[:-1]] + [cf.name[:-1] for cf in counterfactual]

    for i in range(summery.shape[0]):
        axes[i*2].set_title('5%', fontsize=10)
        axes[i*2 + 1].set_title('95%', fontsize=10)

        pos0 = axes[i*2].get_position()
        pos1 = axes[i*2 + 1].get_position()

        # Calculate the center between the two columns
        center_x = (pos0.x0 + pos1.x1) / 2
        # Set the height slightly above the individual titles
        top_y = pos0.y1 + 0.01 

        fig.text(center_x, top_y, titles[i], 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.savefig("figs/attrbution_increase_map" + region + ".png", dpi = 300) 

if __name__=="__main__":
    dir1 = "outputs/outputs_scratch/play/"
    dir2 = "/_16-frac_points_0.02/" 

    region = "Pantanal"

    obs_dir = 'data/data/driving_data_base//'
    obs_file = 'burnt_area_data.csv'  
     
    #plot_all_attribution_scatter(dir1, '/time_series/' +dir2, region, obs_dir, obs_file)
    #plot_confidence_in_attribution(dir1, '/samples/' +  dir2, region)
    plot_change_in_burned_area(dir1, '/samples/' +  dir2, region, obs_dir = obs_dir)

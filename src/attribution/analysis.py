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

if __name__=="__main__":
    dir1 = "outputs/outputs_scratch/play/"
    dir2 = "/time_series/_16-frac_points_0.02/" 

    region = "Pantanal"

    obs_dir = 'data/data/driving_data_base//'
    obs_file = 'burnt_area_data.csv'  
    
    plot_all_attribution_scatter(dir1, dir2, region, obs_dir, obs_file)



    dir2 = "/samples/_16-frac_points_0.02/"
    eg_file  = list(Path(dir1 + get_region_info(region)[region]['dir'] + dir2 + '/').rglob('*.nc'))[0]
    eg_cube = iris.load_cube(eg_file)
    
    nplots = eg_cube.shape[0] + 2
    nrows = int(np.ceil(np.sqrt(nplots)))
    ncols = int(np.ceil(nplots/nrows))
    
    fig, axes = set_up_sow_plot_windows(nrows, ncols, eg_cube[0], size_scale = 3)

    for mnth, ax in zip(eg_cube.coord('month_number').points, axes[:-1]):
        month_idx = '0' + str(mnth + 1) if mnth <9 else str(mnth + 1)#
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
    y1 = pos1.y0  # top of the top row
    height = 0.8  # thickness of the colorbar

    cbar_ax = fig.add_axes([x0, y1 + 0.01, x1 - x0, height]) 
    cbar_ax.set_visible(False)
    add_attribubtion_map_cbar(img, cbar_ax)
    
    plt.savefig("figs/attrbution_where-base" + region + ".png", dpi = 300) 

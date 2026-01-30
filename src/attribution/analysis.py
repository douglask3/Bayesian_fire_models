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

import sys
sys.path.append('.')
sys.path.append('plotting/')
sys.path.append('src/')
sys.path.append('src/attribution/')
sys.path.append('make_input/burnt_area/')
from plot_effect_rr import *
from attribution_where import *
from plot_BA_climateology import *
from plot_change_in_burned_area import *
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
    set_trace()
    plot_attribution_scatter_generic("counterfactual")
    plot_attribution_scatter_generic("counterfactual-extraNoise-")
    plot_attribution_scatter_generic("counterfactual", "rr_line",
                                     effect_ratio_and_rr_over_range)
    plot_attribution_scatter_generic("counterfactual-extraNoise-", "rr_line",
                                     effect_ratio_and_rr_over_range)
    
def attribution_analysis(dir1, dir2, obs_dir, obs_file_nc = "burnt_area.nc",
                         obs_file_csv = 'burnt_area_data.csv', region = ""):
    plot_all_attribution_scatter(dir1, '/time_series/' +dir2, region, obs_dir, obs_file_csv)
    plot_confidence_in_attribution(dir1, '/samples/' +  dir2, region)
    plot_change_in_burned_area(dir1, '/samples/' +  dir2, region, 
                               obs_file = obs_dir + '/' + obs_file_nc)


if __name__=="__main__":
    dir1 = "outputs/outputs_scratch/attribution-base-localBA-data-NEW4/"
    #dir1 = "outputs/outputs_scratch/Amazon-Maria-full-3/"
    dir1 = "outputs/outputs_scratch/Pantanal-Maria-full-3-all-year/"
    dir2 = "/_16-frac_points_0.5/" 
    #dir2 = "/_15-frac_points_0.2/" 

    region = "Pantanal"
    #region = "Amazon"

    obs_dir = 'data/data/driving_data_base//'
    obs_file = 'burnt_area_data.csv'  
     
    

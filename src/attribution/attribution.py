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

def plot_all_attribution_scatter(dir1, dir2, regions, obs_dir, obs_file, *args, **kw):
    
    if not isinstance(regions, list):
        regions = [regions]
    def plot_attribution_scatter_generic(counterfactual_name, 
                                         plot_name = "attribution_scatter",
                                         plot_FUN = plot_fact_vs_counter, flatten = True):
        
        plot_attribution_scatter(regions, plot_name + counterfactual_name,
                                 dir1 = dir1, dir2 = dir2,
                                 obs_dir = obs_dir, obs_file = obs_file, 
                                 plot_FUN = plot_FUN, counterfactual_name = counterfactual_name,
                                 flatten = flatten, *args, **kw)
    path = Path(dir1 + regions[0] + dir2)
    directories = [d for d in path.iterdir() if d.is_dir()] 
    directories = [d.name for d in directories if "counterfactual" in d.name]
    
    for dir in directories:
        #set_trace()
        plot_attribution_scatter_generic(dir)
        #plot_attribution_scatter_generic(dir, "time_series", plot_attribution_time_series,
        #                                False)
        #plot_attribution_scatter_generic("counterfactual-extraNoise-")
        plot_attribution_scatter_generic(dir, "rr_line",
                                         effect_ratio_and_rr_over_range)
    #plot_attribution_scatter_generic("counterfactual-extraNoise-", "rr_line",
    #                                 effect_ratio_and_rr_over_range)
    
def attribution_analysis(dir1, dir2, obs_dir, obs_file_nc = "burned_area.nc",
                         obs_file_csv = 'burned_area.csv', region = "", *args, **kw):

     
    #plot_all_attribution_scatter(dir1, '/time_series/' +dir2, region, obs_dir, obs_file_csv,*args, **kw)
    
    #plot_confidence_in_attribution(dir1, '/samples/' +  dir2, region, *args, **kw)
    
    plot_change_in_burned_area(dir1, '/samples/' +  dir2, region, 
                               obs_file = obs_dir + '/' + region + '/' + obs_file_nc, 
                               counterfactual_name = 'counterfactual-',
                               percentiles = [50], plot_att_only = True, 
                               out_figname_extra = 'season-', *args, **kw)
    return None
    for percentiles in [[50], [5, 95]]:
        try:
            plot_change_in_burned_area(dir1, '/samples/' +  dir2, region, 
                                       obs_file = obs_dir + '/' + region + '/' + obs_file_nc, 
                                       percentiles = percentiles, *args, **kw)
        except:
            plot_change_in_burned_area(dir1, '/samples/' +  dir2, region, 
                                       obs_file = obs_dir + '/' + obs_file_nc, 
                                       percentiles = percentiles, *args, **kw)

    plot_attribution_time_series(dir1, '/samples/' +  dir2, region, *args, **kw)
    

if __name__=="__main__":

    dir1 = "outputs/outputs_scratch/SoW2526/attribution-HadGEM-test29/"
    dir1 = "outputs/outputs_scratch/SoW2526/attribution-HadGEM-test29-fuelcf4/"
    dir1 = "outputs/outputs_scratch/SoW2526/attribution-HadGEM-test29-fuelcf4-rerun2/"
    dir2 = "/_16-frac_points_0.5/" 

    regions = ["Scottish Highlands", "Northwest Iberia", "Midwestern Canadian Shield forests", "Chilean Temperate Forests and Matorral"]
    #regions = "Northwest_Iberia"
    mnthss = [[2, 3, 4, 5,6, 7], [7], [6, 7], [0,1]]#, c('06', '07'), c('03'))
    years = [2025, 2025, 2025, 2026]#, 2025, 2025)

    obs_dir = 'data/data/driving_data2526/'
    obs_file_nc = 'nrt24/factual/burned_area.nc' 
    obs_file_csv = 'nrt24/factual/burned_area.csv'  
    
    for region, mnths, year in zip(regions, mnthss, years):
        attribution_analysis(dir1, dir2, obs_dir, region = region.replace(' ', '_'), year = year, mnths = mnths,
                             obs_file_nc = obs_file_nc, obs_file_csv = obs_file_csv)
     
    

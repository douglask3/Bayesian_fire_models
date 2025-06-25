import sys
sys.path.append('.')
sys.path.append('src/')
sys.path.append('SoW_info/')
from state_of_wildfires_colours  import SoW_cmap
from state_of_wildfires_region_info  import get_region_info
from plot_attribution_scatter import *
from scipy.ndimage import gaussian_filter1d

def effect_ratio_and_rr_over_range(factual_flat, counterfactual_flat, 
                                   obs, plot_name, ax = None):
    effect_ratio = factual_flat/counterfactual_flat
    xs = np.arange(0, np.nanmax(factual_flat)/2, np.nanmax(factual_flat)/1000)
    def for_x(x):
        test = factual_flat > x
        er = np.percentile(effect_ratio[test], [10, 25, 50, 75, 90])
        
        rr = np.sum(test)/np.sum(counterfactual_flat > x)
        return np.append(er, rr)
    outs = np.array([for_x(x) for x in xs])
    if obs > xs.max():
        xs = xs * obs * 1.1/xs.max()
    #set_trace()
    #def filter(ys, sigma = 2):
    #    gaussian_filter1d(ys, sigma)
    outs = scale2upper1(outs)
    outs = gaussian_filter1d(outs, sigma=10.0, axis = 0)
    
    p5, p10, p50, p90, p95 = outs[:,0], outs[:,1], outs[:,2], outs[:,3], outs[:,4]
    risk_ratio = outs[:,5]


    # Plot the median
    ax.plot(xs, p50, color='tab:blue', label='50% percentile')
    
    # Fill between percentiles
    ax.fill_between(xs, p10, p90, color='tab:blue', alpha=0.2, label='25-75%')
    ax.fill_between(xs, p5, p95, color='tab:blue', alpha=0.1, label='10-90%')

    ax.axhline(0.5, color='k', linestyle='--')#, label='No Change (Ratio = 1)')
    ax.axvline(obs, color='red', linestyle='--', label='Observed Burned Area')
    # Left axis labels
    ax.set_xlabel('')
    ax.set_ylabel('', color='tab:blue')
    scale2upper1_axis(ax)
    #ax.tick_params(axis='y')#, labelcolor='tab:blue')
    ax.grid(alpha=0.3)

    # Twin axis for risk ratio
    #ax2 = ax.twinx()
    ax.plot(xs, risk_ratio, color='tab:red', label='Risk ratio', linewidth=2)
    #ax2.set_ylabel('Risk ratio', color='tab:red')
    #ax2.tick_params(axis='y', labelcolor='tab:red')
    
    #scale2upper1_axis(ax2)
    # Legends
    ax.legend(loc='lower left')
    #ax2.legend(loc='lower right')


if __name__=="__main__":
    dir1 = "outputs/outputs_scratch/ConFLAME_nrt-attribution9/"
    dir2 = "-2425/time_series/_19-frac_points_0.5/"

    regions = ["Amazon"]#, "Pantanal"]#"Congo", 
    #retgions = {key: regions_info[key] for key in region_names if key in regions_info}]
    regions = ["Amazon", "Pantanal", "LA", "Congo"]
    region_names = ['Northeast Amazonia', 'Pantanal and Chiquitano', 
                    'Southern California','Congo Basin']
    obs_dir = 'data/data/driving_data2425//'
    obs_file = 'burnt_area_data.csv'

    plot_attribution_scatter(regions, "attribution_metrics_era5_2425",
                             dir1 = dir1, dir2 = dir2,
                             obs_dir = obs_dir, obs_file = obs_file, 
                             plot_FUN = effect_ratio_and_rr_over_range) 

    plot_attribution_scatter(regions, "attribution_metrics_era5_cf_mean_2425",
                             dir1 = dir1, dir2 = dir2, 
                             counterfactual_name = 'counterfactual-metmean',
                             obs_dir = obs_dir, obs_file = obs_file, 
                             plot_FUN = effect_ratio_and_rr_over_range) 
    

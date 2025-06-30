import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pdb import set_trace
import os
import sys
sys.path.append('.')
sys.path.append('src/')
sys.path.append('SoW_info/')
from state_of_wildfires_colours  import SoW_cmap
from state_of_wildfires_region_info  import get_region_info
from plot_attribution_scatter import *

# --- Load data ---
# Model ensemble data

def plot_anaomly_vs_obs(ensemble_matched, obs_ratio_matched, common_times, scale, unit, ax, 
                        color = '#b50000', last12 = True, add_xticks = False):
    # Compute percentiles across the ensemble members
    if scale is not None:
        ensemble_matched = ensemble_matched.mul(scale, axis = 0)
        
    
    def percentile_fun(pc):
        out = np.nanpercentile(ensemble_matched, pc, axis=0)
        if unit == 'ratio':
            out = scale2upper1(out)
        return out

    pcs = [percentile_fun(pc) for pc in [10, 25, 50, 75, 90]]# = np.nanpercentile(ensemble_matched, 10, axis=0)
    #p25 = np.nanpercentile(ensemble_matched, 25, axis=0)
    #p50 = np.nanpercentile(ensemble_matched, 50, axis=0)  # median
    #p75 = np.nanpercentile(ensemble_matched, 75, axis=0)
    #p90 = np.nanpercentile(ensemble_matched, 90, axis=0)

    # Time axis
    #time = np.arange(ensemble_matched.shape[1])
    # Plot shaded regions for model spread
    
    if obs_ratio_matched is not None:
        if unit == 'ratio':
            obs_ratio_matched = scale2upper1(obs_ratio_matched)
        ax.plot(common_times, obs_ratio_matched, color='blue', lw=2, label='Observed')#, marker='o'
    if unit == 'ratio':
        ax.axhline(0.5, color='k', linestyle='--')#, label='No Change (Ratio = 1)')
    ax.fill_between(common_times, pcs[0], pcs[-1], color=color, alpha=0.2, label='10-90% range')
    ax.fill_between(common_times, pcs[1], pcs[-2], color=color, alpha=0.4, label='25-75% range')

    # Plot ensemble median
    ax.plot(common_times, pcs[2], color=color, lw=2, label='Ensemble median')

    # Plot observations
    if obs_ratio_matched is not None:
        ax.plot(common_times, obs_ratio_matched, color='blue', lw=1)#, marker='o'

    # Labels and legend
    
    #ax.set_title('Observed vs Ensemble Model Burned Area')

    if last12:
        end_date = common_times[-1]
        # Get start date 1 year before
        start_date = end_date - pd.DateOffset(years=2)
    
    # Set x-axis limits
        
        ax.set_xlim(start_date, end_date)
        
        mask = (common_times >= start_date) & (common_times <= end_date)
    
        # Get model bounds over that period
        ymin = np.min(pcs[0][mask])
        ymax = np.max(pcs[-1][mask])
        
        if obs_ratio_matched is not None:
            # Also check obs if you want to include that in the range
            obs_min = np.min(obs_ratio_matched[mask])
            obs_max = np.max(obs_ratio_matched[mask])
    
            # Combine
            ymin = min(ymin, obs_min)
            ymax = max(ymax, obs_max)
        
        ax.set_ylim(ymin, ymax)
    #set_trace()
    if unit == 'ratio':
        scale2upper1_axis(ax, ylim = [min(ymin, 1-ymax)])
    ax.grid(True)
    if add_xticks: 
        ax.tick_params(axis='x', labelrotation=45)
    else:
        ax.set_xticklabels([])
    return ensemble_matched


def for_region(dir1, dir2, obs_dir, obs_var, region, region_size,
               controls, contol_names, unit, axes):
    mod_dir = dir1 + region + dir2
    obs_file = obs_dir + region + "/burnt_area_data.csv"
    mod_file = mod_dir + "/" + unit + "/Evaluate.csv"
    if obs_var[-8:] == 'absolute':
        obs_var = obs_var[:-9]
    ensemble_df = pd.read_csv(mod_file)
    ensemble_df.set_index('realization', inplace=True)
    
    # Observation data
    obs_df = pd.read_csv(obs_file)
    obs_df['time'] = pd.to_datetime(obs_df['time'])
    obs_df.set_index('time', inplace=True)    

    # --- Ensure matching time steps ---
    # Parse time columns in ensemble data
    ensemble_df.columns = pd.to_datetime(ensemble_df.columns)
    
    # Find matching time steps
    common_times = ensemble_df.columns.intersection(obs_df.index)
     
    # Subset both datasets to common time steps
    ensemble_matched = ensemble_df[common_times]
    obs_ratio_matched = obs_df.loc[common_times, obs_var]
    if unit == 'anomaly' or unit == 'absolute':
        obs_ratio_matched = region_size * obs_ratio_matched/100.0
        scale = np.mean(np.abs(ensemble_matched), axis = 1)
        scale = np.mean(np.abs(obs_ratio_matched))/scale
        #set_trace()
        #obs_ratio_matched = obs_ratio_matched * region_size
    else:
        scale = None
        region_size = 1.0
    
    dfp = plot_anaomly_vs_obs(ensemble_matched, obs_ratio_matched, common_times, scale, unit, axes[0])
    file_out = mod_dir + "/" + unit + "/Burnt_Area-rescaled4plot.csv"
    dfp.to_csv(file_out)
    percentiles = [10, 25, 50, 75, 90]
    df_percentiles = pd.DataFrame(
        np.percentile(dfp.values, percentiles, axis=0),
        columns=dfp.columns,
        index=[f'p{p}' for p in percentiles]
    )
    file_out = file_out.replace("members", "percentles")
    df_percentiles.to_csv(file_out)

    def for_variable(variable = "ratio/Control.csv"):
        sample_file = mod_dir + variable
        if 'potential_clim'  in variable:
            variable = 'absolute/' + variable.split('/')[1]
        control_df = pd.read_csv(sample_file)
        
        try:
            control_df.set_index(control_df.columns[0], inplace=True)
        except:
            set_trace()
        try:
            control_df.columns = pd.to_datetime(control_df.columns)
        except:
            set_trace()
        #set_trace()
        
        return control_df[common_times]
    
    control_dfs = [for_variable(unit + '/' + file)  for file in controls]
    total =control_dfs[0].copy()
    for df in control_dfs[1:]:
        total += df
    
    if unit == 'anomaly':
        scale = scale * len(controls) * \
            np.max(np.abs(ensemble_matched.values))/np.max(np.abs(total.values))
        scale = region_size/10.0
    elif unit == 'absolute':
        scale = 100.0
    else:
        scale = None
    #set_trace()
    for i, df in enumerate(control_dfs):
        if unit == 'absolute':
            df.iloc[:] /= total.iloc[:] 
        axi = i + 1
        add_xticks = axi == (len(axes)-1)
        dfp = plot_anaomly_vs_obs(df , None, common_times, scale, unit, 
                            axes[axi], colors[i], add_xticks = add_xticks)
        file_out = mod_dir + "/" + unit + "/" + controls[i][:-4] + '-rescaled4plot.csv'
        
        dfp.to_csv(file_out)
        percentiles = [10, 25, 50, 75, 90]
        df_percentiles = pd.DataFrame(
            np.percentile(dfp.values, percentiles, axis=0),
            columns=dfp.columns,
            index=[f'p{p}' for p in percentiles]
        )
        file_out = file_out.replace("members", "percentles")
        df_percentiles.to_csv(file_out)
        
        if region == regions[0]:
            if unit == 'absolute':
                axes[0].set_ylabel('Burned Area (km$^2$)')
            else:
                axes[0].set_ylabel('Burned Area')
            axes[i+1].set_ylabel(contol_names[i])
            


dir1 = "outputs/outputs_scratch/ConFLAME_nrt-drivers8/" 
dir2 = ["-2425/time_series/_21-frac_points_0.5/baseline-/mean/members/",
        "-2425/time_series/_21-frac_points_0.5/baseline-/pc-95.0/members/"]
obs_dir = "data/data/driving_data2425/"
regions = ['Amazon', 'Pantanal', 'LA',  'Congo']
region_names = ['Northeast Amazonia', 'Pantanal and Chiquitano', 
                    'Southern California','Congo Basin']
regions = ['Amazon', 'Pantanal', 'LA',  'Congo']
region_sizes = [3592.46844, 1063.64005, 95.11927, 2678.12835]
controls = ['standard-Fuel.csv', 'standard-Moisture.csv', 'standard-Weather.csv', 'standard-Wind.csv', 'standard-Ignition.csv', 'standard-Suppression.csv']
controls = ['potential_climateology-Fuel.csv', 'potential_climateology-Moisture.csv', 
            'potential_climateology-Weather.csv', 'potential_climateology-Wind.csv', 
            'potential_climateology-Ignition.csv', 'potential_climateology-Suppression.csv']
contol_names = ['Fuel', 'Moisture', 'Weather', 'Wind', 'Ignition', 'Suppression']
colors = ['#e98400', '#e98400', '#ee007f', '#ee007f', '#0096a1', '#0096a1']

def run_regions_controls(regions, controls, contol_names, unit = 'ratio'):
    fig, axes = plt.subplots(len(controls) + 1, len(regions), figsize=(4*len(regions), 1.53*(len(controls) + 1)))
    
    for i, region in enumerate(regions):
        print(i)
        if len(regions) == 1:
            axesi = axes
        else:
            axesi = axes[:, i] 
        for_region(dir1, dir2[0], obs_dir, 'mean_burnt_area_' + unit, region, region_sizes[i],
                   controls, contol_names, unit, axesi)
        
        if len(regions) == 1:
            axesi[0].set_title('Burned Area Anomoly')
        else:
            axesi[0].set_title(region_names[i])
    #if i == 0:
    #    axesi[0].set_title('All region')
    
    #for_region(dir1, dir2[1], obs_dir, 'p95_burnt_ratio', region, axesi[1])
    #if i == 0:
    #    axesi[1].set_title('Highest Burned Areas')

#plt.legend(ncol = 2)
        plt.tight_layout()
    fig.subplots_adjust(left=0.09)
    if unit == 'ratio':
        fig.text(0.02  , 0.5,  r'Control Contribution (ratio)', va='center', rotation='vertical')
    elif unit == 'anomaly':
        fig.text(0.02, 0.5,  r'Control Contribution (km$^2$)', va='center', rotation='vertical')
    else:
        #fig.text(0.02, 0.1,  r'Control Influence (%)', va='center', rotation='vertical')
        fig.text(0.02, 0.4,  r'Control Influence (%)', va='center', rotation='vertical')
        
    plt.savefig('figs/BA_anaomoly-' + str(len(regions)) + '-' + str(len(controls)) + \
                '-' + unit + '.png', 
                dpi = 300)
    plt.savefig('figs/BA_anaomoly-' + str(len(regions)) + '-' + str(len(controls)) + \
                '-' + unit + '.pdf')
#plt.show()
#set_trace()

controls = ['potential_climateology-Fuel-Moisture.csv', 'potential_climateology-Weather-Wind.csv', 'potential_climateology-Suppression-Ignition.csv']
contol_names = ['Fuel', 'Weather', 'Ignitions/Humans']
colors = ['#e98400', '#0096a1','#ee007f']
run_regions_controls(regions, controls, contol_names, 'anomaly')

dir1 = "outputs/outputs_scratch/ConFLAME_nrt-drivers6/" 
controls = ['standard-Fuel-Moisture.csv', 'standard-Weather-Wind.csv', 'standard-Suppression-Ignition.csv']
run_regions_controls(regions, controls, contol_names, 'absolute')




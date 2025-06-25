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

def plot_anaomly_vs_obs(ensemble_matched, obs_ratio_matched, common_times,ax, 
                        color = '#b50000', last12 = True):
    # Compute percentiles across the ensemble members
    def percentile_fun(pc):
        return scale2upper1(np.nanpercentile(ensemble_matched, pc, axis=0))
    pcs = [percentile_fun(pc) for pc in [10, 25, 50, 75, 90]]# = np.nanpercentile(ensemble_matched, 10, axis=0)
    #p25 = np.nanpercentile(ensemble_matched, 25, axis=0)
    #p50 = np.nanpercentile(ensemble_matched, 50, axis=0)  # median
    #p75 = np.nanpercentile(ensemble_matched, 75, axis=0)
    #p90 = np.nanpercentile(ensemble_matched, 90, axis=0)

    # Time axis
    #time = np.arange(ensemble_matched.shape[1])
    # Plot shaded regions for model spread
    
    if obs_ratio_matched is not None:
        obs_ratio_matched = scale2upper1(obs_ratio_matched)
        ax.plot(common_times, obs_ratio_matched, color='blue', lw=2, label='Observed')#, marker='o'
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
    scale2upper1_axis(ax, ylim = [min(ymin, 1-ymax)])
    ax.grid(True)

def for_region(dir1, dir2, obs_dir, obs_var, region, controls, axes):
    mod_dir = dir1 + region + dir2
    obs_file = obs_dir + region + "/burnt_area_data.csv"
    mod_file = mod_dir + "/ratio/Evaluate.csv"
    
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
    plot_anaomly_vs_obs(ensemble_matched, obs_ratio_matched, common_times, axes[0])
    # --- Log-transform (add small epsilon to avoid log(0)) ---
    epsilon = 1e-8
    log_ensemble = np.log(ensemble_matched + epsilon)
    log_obs = np.log(obs_ratio_matched + epsilon)
    
    # --- Compute squared distance for each ensemble member ---
    # Result is a Series with one entry per ensemble member
    squared_distances = ((log_ensemble.subtract(log_obs, axis=1))**2).sum(axis=1)
    
    sigma = np.std(squared_distances)
    gaussian_weights = np.exp(-squared_distances / (2 * sigma**2))
    weights = gaussian_weights/np.sum(gaussian_weights)
    
    def for_variable(variable = "ratio/Control.csv", axi = 0, color = 'blue'):
        sample_file = mod_dir + variable

        control_df = pd.read_csv(sample_file)
        try:
            control_df.set_index(control_df.columns[0], inplace=True)
        except:
            set_trace()
        control_df.columns = pd.to_datetime(control_df.columns)
        #set_trace()
        plot_anaomly_vs_obs(control_df[common_times], None, common_times, axes[axi], color)
        # --- Weighted bootstrap from Control, time step by time step ---
        N, M = control_df.shape
        realizations = control_df.index.to_numpy()
        bootstrapped_array = np.empty((N, M))
        
        rng = np.random.default_rng(seed=42)  # for reproducibility
        
        for i, time in enumerate(control_df.columns):
            try:
                sampled_idxs = rng.choice(len(realizations), size=N, replace=True, p=weights.values)
            except:
                set_trace()
            bootstrapped_array[:, i] = control_df.iloc[sampled_idxs, i].values
        
        # Optional: wrap into DataFrame with the same index/columns as control_matched
        bootstrapped_df = pd.DataFrame(bootstrapped_array, columns=control_df.columns)
        bootstrapped_df.index = range(N)  # or use sampled_idxs if you want source index info
        #bootstrapped_df.to_csv(sample_file[:-4] + '-weighted.csv')
    
    #dirs = ["absolute", "anomaly", "ratio"]
    dir = "ratio"
    #for dir in dirs:
    #files = os.listdir(mod_dir + dir + '/')
    #set_trace() 
    colors = ['#e98400', '#e98400', '#ee007f', '#ee007f', '#0096a1', '#0096a1']
    for i, file in enumerate(controls):
        #if 'csv' in file and 'standard' in file and 'weighted' not in files[3]:
        for_variable(dir + '/' + file, i + 1, colors[i])
        if region == regions[0]:
            axes[i+1].set_ylabel(file.split('-')[1].split('.')[0])


dir1 = "outputs/outputs_scratch/ConFLAME_nrt-drivers6/" 
dir2 = ["-2425/time_series/_21-frac_points_0.5/baseline-/mean/members/",
        "-2425/time_series/_21-frac_points_0.5/baseline-/pc-95.0/members/"]
obs_dir = "data/data/driving_data2425/"
regions = ['Amazon', 'Pantanal', 'LA',  'Congo']
region_names = ['Northeast Amazonia', 'Pantanal and Chiquitano', 
                    'Southern California','Congo Basin']
regions = ['Amazon', 'Pantanal']

controls = ['standard-Fuel.csv', 'standard-Moisture.csv', 'standard-Weather.csv', 'standard-Wind.csv', 'standard-Ignition.csv', 'standard-Suppression.csv']
fig, axes = plt.subplots(len(controls) + 1, len(regions), figsize=(7*len(regions), 2.67*(len(controls) + 1)))

for i, region in enumerate(regions):
    print(i)
    if len(regions) == 1:
        axesi = axes
    else:
        axesi = axes[:, i]
        
    for_region(dir1, dir2[0], obs_dir, 'mean_burnt_area_ratio', region, controls, axesi)
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
plt.savefig('figs/BA_anaomoly' + str(len(regions)) + '.png', dpi = 300)
#plt.show()
#set_trace()

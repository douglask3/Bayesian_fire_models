import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pdb import set_trace
import os
# --- Load data ---
# Model ensemble data

def plot_anaomly_vs_obs(ensemble_matched, obs_ratio_matched, common_times,ax, last12 = False):
    # Compute percentiles across the ensemble members
    p10 = np.nanpercentile(ensemble_matched, 10, axis=0)
    p25 = np.nanpercentile(ensemble_matched, 25, axis=0)
    p50 = np.nanpercentile(ensemble_matched, 50, axis=0)  # median
    p75 = np.nanpercentile(ensemble_matched, 75, axis=0)
    p90 = np.nanpercentile(ensemble_matched, 90, axis=0)

    # Time axis
    #time = np.arange(ensemble_matched.shape[1])
    # Plot shaded regions for model spread
    ax.plot(common_times, obs_ratio_matched, color='red', lw=1, label='Observed')#, marker='o'

    ax.fill_between(common_times, p10, p90, color='lightblue', alpha=0.4, label='10-90% range')
    ax.fill_between(common_times, p25, p75, color='dodgerblue', alpha=0.6, label='25-75% range')

    # Plot ensemble median
    ax.plot(common_times, p50, color='blue', lw=2, label='Ensemble median')

    # Plot observations
    ax.plot(common_times, obs_ratio_matched, color='red', lw=0.5)#, marker='o'

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
        model_min = np.min(p10[mask])
        model_max = np.max(p90[mask])
        
        # Also check obs if you want to include that in the range
        obs_min = np.min(obs_ratio_matched[mask])
        obs_max = np.max(obs_ratio_matched[mask])
    
        # Combine
        ymin = min(model_min, obs_min)
        ymax = max(model_max, obs_max)
        ax.set_ylim(ymin, ymax)
        
    ax.grid(True)

def for_region(dir1, dir2, obs_dir, obs_var, region, ax):
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
    plot_anaomly_vs_obs(ensemble_matched, obs_ratio_matched, common_times, ax)
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
    
    def for_variable(variable = "ratio/Control.csv"):
        sample_file = mod_dir + variable

        control_df = pd.read_csv(sample_file)
        try:
            control_df.set_index(control_df.columns[0], inplace=True)
        except:
            set_trace()
        #control_df.columns = pd.to_datetime(control_df.columns)

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
        bootstrapped_df.to_csv(sample_file[:-4] + '-weighted.csv')
    
    dirs = ["absolute", "anomaly", "ratio"]
    for dir in dirs:
        files = os.listdir(mod_dir + dir + '/')
        for file in files:
            if 'csv' in file and 'standard' in file and 'weighted' not in files[3]:
                for_variable(dir + '/' + file)


dir1 = "outputs/outputs_scratch/ConFLAME_nrt-drivers6/" 
dir2 = ["-2425/time_series/_21-frac_points_0.5/baseline-/mean/members/",
        "-2425/time_series/_21-frac_points_0.5/baseline-/pc-95.0/members/"]
obs_dir = "data/data/driving_data2425/"
regions = ['Amazon', 'Pantanal', 'LA',  'Congo']
region_names = ['Northeast Amazonia', 'Pantanal and Chiquitano', 
                    'Southern California','Congo Basin']
regions = ['Amazon']

fig, axes = plt.subplots(len(regions), 2, figsize=(10, 2*len(regions)))

for i, region in enumerate(regions):
    print(i)
    if len(regions) == 1:
        axesi = axes
    else:
        axesi = axes[i,:]
        
    for_region(dir1, dir2[0], obs_dir, 'mean_burnt_area_ratio', region, axesi[0])
    if len(regions) == 1:
        axesi[0].set_ylabel('Burned Area Anomoly')
    else:
        axesi[0].set_ylabel(region_names[i])
    if i == 0:
        axesi[0].set_title('All region')
    
    for_region(dir1, dir2[1], obs_dir, 'p95_burnt_ratio', region, axesi[1])
    if i == 0:
        axesi[1].set_title('Highest Burned Areas')

plt.legend(ncol = 2)
plt.tight_layout()
plt.savefig('figs/BA_anaomoly' + str(len(regions)) + '.png', dpi = 300)
#plt.show()

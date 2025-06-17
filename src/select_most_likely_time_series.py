import pandas as pd
import numpy as np
from pdb import set_trace

# --- Load data ---
# Model ensemble data
mod_dir = "outputs/outputs_scratch/ConFLAME_nrt-drivers4/Amazon-2425/time_series/_21-frac_points_0.5/baseline-/mean/members/ratio/"
mod_file = mod_dir + "Evaluate.csv"
sample_file = mod_dir + "Control.csv"
obs_file = 'data/data/driving_data2425/Amazon/burnt_area_data.csv'

control_df = pd.read_csv(sample_file)
control_df.set_index('realization', inplace=True)
control_df.columns = pd.to_datetime(control_df.columns)

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
control_matched = control_df[common_times]
obs_ratio_matched = obs_df.loc[common_times, 'mean_burnt_area_ratio']

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

print(weights)

# --- Weighted bootstrap from Control, time step by time step ---
N, M = control_matched.shape
realizations = control_matched.index.to_numpy()
bootstrapped_array = np.empty((N, M))

rng = np.random.default_rng(seed=42)  # for reproducibility

for i, time in enumerate(control_matched.columns):
    sampled_idxs = rng.choice(len(realizations), size=N, replace=True, p=weights.values)
    bootstrapped_array[:, i] = control_matched.iloc[sampled_idxs, i].values

# Optional: wrap into DataFrame with the same index/columns as control_matched
bootstrapped_df = pd.DataFrame(bootstrapped_array, columns=control_matched.columns)
bootstrapped_df.index = range(N)  # or use sampled_idxs if you want source index info

print(bootstrapped_df.head())



from pdb import set_trace
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
factual = pd.read_csv("outputs/outputs/ConFire_nrt2_Amazon-2425/figs/_13-frac_points_0.5-factual-control_TS/mean/points-Evaluate.csv", header=None)
counterfactual = pd.read_csv("outputs/outputs/ConFire_nrt2_Amazon-2425/figs/_13-frac_points_0.5-counterfactual-control_TS/mean/points-Evaluate.csv", header=None)
ix = [i for i in range(factual.shape[1]) if i % 12 in [1, 2, 3]]
# Flatten the arrays to 1D
factual_flat = factual.values[:,ix].flatten()*100
counterfactual_flat = counterfactual.values[:,ix].flatten()*100 #[:,12:]
set_trace()

# Scatter plot 1: Factual vs Counterfactual
plt.figure(figsize=(6, 6))
plt.scatter(factual_flat, counterfactual_flat, alpha=0.3, s=5)
plt.plot([0, factual_flat.max()], [0, factual_flat.max()], 'k--', label='1:1 Line')
plt.ylabel("Counterfactual Burned Area")
plt.xlabel("Factual Burned Area")
plt.title("Factual vs Counterfactual Burned Area")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()

# Avoid division by zero
counterfactual_flat_safe = counterfactual_flat.copy()
#counterfactual_flat_safe[counterfactual_flat_safe == 0] = 1e-16

# Scatter plot 2: Factual vs Factual/Counterfactual
effect_ratio = (factual_flat - counterfactual_flat)*100
effect_ratio[effect_ratio>0] = effect_ratio[effect_ratio>0]/factual_flat[effect_ratio>0]
effect_ratio[effect_ratio<0] = effect_ratio[effect_ratio<0]/counterfactual_flat[effect_ratio<0]

plt.figure(figsize=(6, 6))
plt.scatter(factual_flat, effect_ratio, alpha=0.3, s=5, color = 'red')
plt.axhline(0, color='k', linestyle='--')#, label='No Change (Ratio = 1)')

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Load NetCDF
ds = xr.open_dataset("data/data/driving_data2425/Amazon/burnt_area.nc")
ba = ds['variable']  # Adjust if your variable has a different name

# If there's a 'time' coordinate with datetime info
if 'time' in ba.coords and np.issubdtype(ba['time'].dtype, np.datetime64):
    jfm_ba = ba.sel(time=ba['time.month'].isin([1, 2, 3]))
else:
    # fallback if months aren't datetime-labeled, assume ordered months
    months = ba.shape[0]  # or len(ba['time'])
    years = months // 12
    ba_reshaped = ba[:years*12].values.reshape((years, 12, -1)) if ba.ndim == 2 else ba[:years*12].values.reshape((years, 12))
    jfm_ba = ba_reshaped[:, 0:3, ...]
    jfm_ba = jfm_ba.max(axis=1)*10  # mean over JFM

# Mean burned area during JFM
set_trace()
jfm_mean_value = jfm_ba.mean(axis = (1,2))[-3:].mean().item()*100  # Get scalar
plt.axvline(jfm_mean_value, color='red', linestyle='--', label='Observed Burned Area')

mask = factual_flat > jfm_mean_value
percentile = [5, 10, 50, 90, 95]
cc_effect = np.percentile(effect_ratio[mask], percentile)/100.0
cc_effect = np.round(1.0+cc_effect/(1.0-cc_effect), 2)
rr = np.mean(effect_ratio[mask]>0)
pv = np.round(rr, 2)
rr = np.round(rr/(1-rr), 2)
percentile_text = [str(pc) + "%:\n" + str(cc) for pc, cc in zip(percentile, cc_effect)]

plt.text(45, -60, "Climate change impact (pvalue: " + str(pv) + ")")
for i in range(len(percentile_text)): plt.text(45 + i*10, -75, percentile_text[i])
plt.text(45, -92, "Risk Ratio:")
plt.text(45, -100, rr)

plt.xlabel("Factual Burned Area")
plt.ylabel("Factual / Counterfactual Ratio")
plt.title("Factual vs Climate Change Effect on Burned Area")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()


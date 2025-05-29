import xarray as xr
from pdb import set_trace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import state_of_wildfires_colours
import seaborn as sns
import sys
sys.path.append('.')
sys.path.append('src/')
from state_of_wildfires_colours  import SoW_cmap

# Load the data
dir1 = "outputs/outputs/ConFire_nrt4"
dir2 = "-2425-fuel2/time_series/_19-frac_points_0.5/"
metric = "mean"
region = "Amazon"
mnths = ['01', '02', '03']
years = [2024]

def flatten(xss):
    return [x for xs in xss for x in xs]

def plot_kde(x, y, xlab, ylab, cmap_name = "gradient_hues_extended", *args, **kw):
    df = pd.DataFrame({xlab: x, ylab: y})
    sns.kdeplot(data=df, x=xlab, y=ylab, fill=True, 
                cmap=SoW_cmap[cmap_name], *args, **kw)

dir = dir1 + region + dir2 + '/'
factual = pd.read_csv(dir + "factual-/" + metric + "/points-Evaluate.csv")
counterfactual = pd.read_csv(dir + "counterfactual-/" + metric + "/points-Evaluate.csv")

def extract_years(df, years, mnths):
    target_cols = [
        f"{year}-{month}-01T00:00:00" 
        for year in years 
        for month in mnths
        if f"{year}-{month}-01T00:00:00" in df.columns
    ]
    
    # Reshape: group columns by year
    avg_per_year = []
    for year in years:
        cols_this_year = [
            f"{year}-{month}-01T00:00:00" 
            for month in mnths 
            if f"{year}-{month}-01T00:00:00" in df.columns
        ]
        avg_per_year.append(df[cols_this_year].mean(axis=1))

    # Convert to final DataFrame
    #set_trace()
    return np.array(avg_per_year).flatten()

# Flatten the arrays to 1D
factual_flat = extract_years(factual, years, mnths) + 0.000000001
counterfactual_flat = extract_years(counterfactual, years, mnths) + 0.000000001
set_trace()
# Scatter plot 1: Factual vs Counterfactual
plt.figure(figsize=(6, 6))

#log_levels = np.geomspace(1e-10, 1.0, 500)
# More bias to lower densities (power < 1)
x = np.linspace(0, 1, 20)
log_levels = x**(8)  # try 3, 5, 7 for increasingly strong bias
plot_kde(factual_flat, counterfactual_flat, "factual", "counterfactual",
         levels=log_levels, log_scale = True, thresh=1e-4)

plt.plot([0.0000000000001, 100], [0.0000000000001, 100], 'k--', label='1:1 Line')
plt.ylabel("Counterfactual Burned Area")
plt.xlabel("Factual Burned Area")
plt.title("Factual vs Counterfactual Burned Area")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Avoid division by zero
counterfactual_flat_safe = counterfactual_flat.copy()

# Scatter plot 2: Factual vs Factual/Counterfactual
effect_ratio = (factual_flat - counterfactual_flat)*100
effect_ratio[effect_ratio>0] = effect_ratio[effect_ratio>0]/factual_flat[effect_ratio>0]
effect_ratio[effect_ratio<0] = effect_ratio[effect_ratio<0]/counterfactual_flat[effect_ratio<0]



# Load NetCDF
ds = xr.open_dataset("data/data/driving_data2425/Amazon/burnt_area.nc")
ba = ds['variable']  # Adjust if your variable has a different name

# If there's a 'time' coordinate with datetime info
if 'time' in ba.coords and np.issubdtype(ba['time'].dtype, np.datetime64):
    jfm_ba = ba.sel(time=ba['time.month'].isin([0, 1, 2]))
else:
    # fallback if months aren't datetime-labeled, assume ordered months
    months = ba.shape[0]  # or len(ba['time'])
    years = months // 12
    ba_reshaped = ba[:years*12].values.reshape((years, 12, -1)) if ba.ndim == 2 else ba[:years*12].values.reshape((years, 12))
    jfm_ba = ba_reshaped[:, 0:3, ...]
    jfm_ba = jfm_ba.max(axis=1)*10  # mean over JFM

# Mean burned area during JFM
jfm_mean_value = jfm_ba.mean(axis = (1,2))[-3:].mean().item()#*100  # Get scalar

plt.figure(figsize=(6, 6))
x = np.linspace(0, 1, 20)
log_levels = x**(5)  # try 3, 5, 7 for increasingly strong bias
#log_levels = np.append(log_levels, np.array([0.5, 1.0]))
xmin = factual_flat.max()/100
test = factual_flat > xmin
#plot_kde(factual_flat, effect_ratio, "factual", "effect ratio",
#         levels=log_levels, thresh=1e-4, bw_adjust = 2, 
#         clip=((0, jfm_mean_value), (-100, 100))) #, log_scale = True
plot_kde(factual_flat[test], effect_ratio[test], "factual", "effect ratio",bw_adjust = 1,
         levels=log_levels, thresh=1e-4, clip=((0.0,factual_flat.max()*1.3), (-100, 100))) #,  = True

plot_kde(factual_flat[test], effect_ratio[test], "factual", "effect ratio",bw_adjust = 1,
         levels=log_levels, thresh=1e-4, clip=((jfm_mean_value,factual_flat.max()*1.3), (-100, 100)))
plt.axhline(0, color='k', linestyle='--')#, label='No Change (Ratio = 1)')



plt.axvline(jfm_mean_value, color='red', linestyle='--', label='Observed Burned Area')

mask = factual_flat > jfm_mean_value
percentile = [10, 50, 90]
cc_effect = np.percentile(effect_ratio[mask], percentile)/100.0
cc_effect = np.round(1.0+cc_effect/(1.0-cc_effect), 2)
rr = np.mean(effect_ratio[mask]>0)
pv = np.round(rr, 2)
rr = np.round(rr/(1-rr), 2)
percentile_text = [str(pc) + "%:\n" + str(cc) for pc, cc in zip(percentile, cc_effect)]

ax = plt.gca()
plt.text(0.55, 0.23, "Climate change impact (pvalue: " + str(pv) + ")", transform=ax.transAxes)
for i in range(len(percentile_text)): plt.text(0.55 + i*0.10, 0.13, percentile_text[i], transform=ax.transAxes)
plt.text(0.55, 0.09, "Risk Ratio:", transform=ax.transAxes)
plt.text(0.55, 0.05, rr, transform=ax.transAxes)

plt.xlabel("Factual Burned Area")
plt.ylabel("Factual / Counterfactual Ratio")
plt.title("Factual vs Climate Change Effect on Burned Area")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()


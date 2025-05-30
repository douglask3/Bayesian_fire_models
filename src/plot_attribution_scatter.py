import xarray as xr
from pdb import set_trace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import state_of_wildfires_colours
import seaborn as sns
import fnmatch
import sys
sys.path.append('.')
sys.path.append('src/')
from state_of_wildfires_colours  import SoW_cmap

def extract_years(df, years, mnths, ext = "-01T00:00:00"):
    if years is None:
        years = np.unique([col[0:4] for col in df.columns[1:]])
    # Reshape: group columns by year
    avg_per_year = []
    for year in years:
        cols_this_year = [
            col for col in df.columns
            for month in mnths
            if fnmatch.fnmatch(col, f"{year}-{month}*")
        ]
    
        avg_per_year.append(df[cols_this_year].mean(axis=1))
    
    return np.array(avg_per_year).flatten()

def flatten(xss):
    return [x for xs in xss for x in xs]

def plot_kde(x, y, xlab, ylab, cmap_name = "gradient_hues_extended", *args, **kw):
    df = pd.DataFrame({xlab: x, ylab: y})
    sns.kdeplot(data=df, x=xlab, y=ylab, fill=True, 
                cmap=SoW_cmap[cmap_name], *args, **kw)

def plot_fact_vs_counter(factual_flat, counterfactual_flat, obs, ax = False): 
    # Scatter plot 1: Factual vs Counterfactual 
    x = np.linspace(0, 1, 20)
    log_levels = x**(8)  # try 3, 5, 7 for increasingly strong bias
    plot_kde(factual_flat, counterfactual_flat, "factual", "counterfactual",
             levels=log_levels, log_scale = True, thresh=1e-4, axes = ax)

    plt.plot([0.0000000001, 100], [0.0000000001, 100], 'k--', label='1:1 Line')
    plt.ylabel("Counterfactual Burned Area")
    plt.xlabel("Factual Burned Area")
    plt.title("Factual vs Counterfactual Burned Area")
    
    plt.axvline(obs, color='red', linestyle='--', label='Observed Burned Area')
    plt.ahvline(obs, color='red', linestyle='--', label='Observed Burned Area')
    
    plt.grid(True)

def plot_fact_vs_ratio(factual_flat, counterfactual_flat, obs, plot_name, ax = None):
    # Scatter plot 2: Factual vs Factual/Counterfactual
    effect_ratio = (factual_flat - counterfactual_flat)*100
    test = effect_ratio>0
    effect_ratio[test] = effect_ratio[test]/factual_flat[test]
    test = ~test
    effect_ratio[test] = effect_ratio[test]/counterfactual_flat[test]

    x = np.linspace(0, 1, 20)
    log_levels = x**(5)  # try 3, 5, 7 for increasingly strong bias
    
    xmin = factual_flat.max()/100
    test = factual_flat > xmin
    
    plot_kde(factual_flat[test], effect_ratio[test], "factual", "effect ratio",bw_adjust = 1,
             levels=log_levels, thresh=1e-4, clip=((0.0,factual_flat.max()), (-100, 100)),
             ax = ax) 
    
    plot_kde(factual_flat[test], effect_ratio[test], "factual", "effect ratio",bw_adjust = 1,
             levels=log_levels, thresh=1e-4, clip=((obs,factual_flat.max()), (-100, 100)),
             ax = ax)
    ax.axhline(0, color='k', linestyle='--')#, label='No Change (Ratio = 1)')

    ax.axvline(obs, color='red', linestyle='--', label='Observed Burned Area')
    
    mask = factual_flat > obs
    percentile = [10, 50, 90]
    try:
        cc_effect = np.percentile(effect_ratio[mask], percentile)/100.0
    except:
        set_trace()
    cc_effect = np.round(1.0+cc_effect/(1.0-cc_effect), 2)
    pv = np.mean(effect_ratio[mask]>0)
    pv = np.round(pv, 2)
    if (pv > 0.99):
        pv = str("> 0.99")
    rr = np.sum(factual_flat>obs)/np.sum(counterfactual_flat > obs)
    rr = np.round(rr, 2)
    percentile_text = [str(pc) + "%:\n" + str(cc) for pc, cc in zip(percentile, cc_effect)]

    #ax = plt.gca()
    ax.text(0.3, 0.35, "Climate change impact (pvalue: " + str(pv) + ")", 
              transform=ax.transAxes, axes = ax)
    
    for i in range(len(percentile_text)):
        ax.text(0.3 + i*0.18, 0.2, percentile_text[i], transform=ax.transAxes)
    ax.text(0.3, 0.09, "Risk Ratio:", transform=ax.transAxes)
    ax.text(0.3, 0.02, rr, transform=ax.transAxes)
    #ax.set_xlabel("Factual burned area (%)")
    #ax.set_ylabel("Relative difference in burned area (%)")
    ax.set_xlabel(" ")
    ax.set_ylabel(plot_name)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    return effect_ratio[mask]



def plot_for_region(region, mnths, years, metric, plot_FUN, 
                    dir1, dir2, obs_dir, obs_file, 
                    all_mod_years = False, add_legend = False,
                    *args, **kw):
    # Load the data
    dir = dir1 + region + dir2 + '/'
    factual = pd.read_csv(dir + "factual-/" + metric + "/points-Evaluate.csv")
    counterfactual = pd.read_csv(dir + "counterfactual-/" + metric + "/points-Evaluate.csv")
    obs = pd.read_csv(obs_dir + '/' + region + '/' + obs_file)

    # Extra years and flatten the arrays to 1D
    if all_mod_years:
        mod_years = None
    else:
        mod_years = years
    factual_flat = extract_years(factual, mod_years, mnths) + 0.000000001
    counterfactual_flat = extract_years(counterfactual, mod_years, mnths) + 0.000000001
    
    obs = extract_years(obs.set_index('time').T, years, mnths, '-15')
    if metric == 'mean':
        obs = obs[0]
        plot_name = region
    else:
        obs = obs[1]
        factual_flat = factual_flat * 100.0
        counterfactual_flat = counterfactual_flat * 100.0
        plot_name = ""
    if obs > factual_flat.max():
        factual_flat = factual_flat * 100
        counterfactual_flat = counterfactual_flat * 100
    out = plot_FUN(factual_flat, counterfactual_flat, obs, plot_name = plot_name, *args, **kw)
    
    if add_legend:
        plt.legend()

    return out

def plot_attribution_scatter(regions, mnths, years, figname, *args, **kw):
    metrics = ["mean", 'pc-95.0']
    fig, axes = plt.subplots(len(regions), 2, figsize=(12, len(regions)*3.5))
    out = []
    for i, metric in  enumerate(metrics):
        outi = []
        for j, region, mnth, yrs in zip (range(len(regions)), regions, mnths, years):
            ax = axes[j, i]
            print(region)
            outi.append(plot_for_region(region, mnth, yrs, metric, plot_fact_vs_ratio, ax = ax, 
                            *args, **kw))
        out.append(outi)
    
    fig.text(0.05, 0.5, "Relative difference in burned area (%)", ha='right', va='center', fontsize=12, rotation=90)
    fig.text(0.23, 0.95, "Entire region", ha='center', va='bottom', fontsize=14)
    fig.text(0.73, 0.95, "High burnt areas", ha='center', va='bottom', fontsize=14)
    fig.text(0.5, 0.05, "Factual burned area (%)", ha='center', va='top', fontsize=12)
     #ax.set_xlabel("Factual burned area (%)")
    #ax.set_ylabel("Relative difference in burned area (%)")
    plt.savefig('figs/' + figname + ".png")
    return out

if __name__=="__main__":
    dir1 = "outputs/outputs/ConFLAME_nrt"
    dir2 = "-2425/time_series/_19-frac_points_0.5/"

    regions = ["Amazon", "Pantanal", "Congo"]
    mnths = [['01', '02', '03'], ['01'], ['07']]
    years = [[2024], [2024], [2024]]
    
    obs_dir = 'data/data/driving_data2425//'
    obs_file = 'burnt_area_data.csv'
    
    outs_era5 = plot_attribution_scatter(regions, mnths, years, "attribution_scatter_era5_2425",
                             dir1 = dir1, dir2 = dir2,
                             obs_dir = obs_dir, obs_file = obs_file) 
    
    dir1 = "outputs/outputs/ConFLAME_"
    dir2 = "-2425-attempt12/time_series/_15-frac_points_0.5/"

    outs_isimip = plot_attribution_scatter(regions, mnths, years, 
                             "attribution_scatter_isimip_2425",
                             dir1 = dir1, dir2 = dir2,
                             obs_dir = obs_dir, obs_file = obs_file, all_mod_years = True) 

    outs_isimip = plot_attribution_scatter(regions, mnths, years, 
                             "attribution_scatter_isimip_2425",
                             dir1 = dir1, dir2 = dir2,
                             obs_dir = obs_dir, obs_file = obs_file, all_mod_years = True) 

    set_trace()

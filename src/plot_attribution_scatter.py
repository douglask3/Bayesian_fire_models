import xarray as xr
from pdb import set_trace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fnmatch
import sys
import pickle
sys.path.append('.')
sys.path.append('src/')
sys.path.append('SoW_info/')
from state_of_wildfires_colours  import SoW_cmap
from state_of_wildfires_region_info  import get_region_info

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
    set_trace()
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



def plot_for_region(region, metric, plot_FUN, 
                    dir1, dir2, obs_dir, obs_file, 
                    factual_name = "factual", counterfactual_name = "counterfactual",
                    all_mod_years = False, add_legend = False,
                    *args, **kw):
    region_info = get_region_info(region)[region]
    
    years = region_info['years']
    mnths = region_info['mnths']
    # Load the data
    dir = dir1 + region + dir2 + '/'
    factual = pd.read_csv(dir + factual_name + "-/" + metric + "/points-Evaluate.csv")
    counterfactual = pd.read_csv(dir + counterfactual_name + \
                                 "-/" + metric + "/points-Evaluate.csv")
    obs = pd.read_csv(obs_dir + '/' + region + '/' + obs_file)

    # Extra years and flatten the arrays to 1D
    if all_mod_years:
        mod_years = None
    else:
        mod_years = years
    factual_flat = extract_years(factual, mod_years, mnths) + 0.000000001
    counterfactual_flat = extract_years(counterfactual, mod_years, mnths) + 0.000000001
    
    obs = extract_years(obs.set_index('time').T, years, mnths, '-15')
    factual_flat0 = factual_flat.copy()
    if metric == 'mean':
        obs = obs[0]
        plot_name = region_info['shortname']
    else:
        obs = obs[1]
        if factual_flat.max() < 1:
            factual_flat = factual_flat * 100.0
            counterfactual_flat = counterfactual_flat * 100.0
        plot_name = ""
    if obs > factual_flat.max() and factual_flat.max() < 1:
        factual_flat = factual_flat * 100
        counterfactual_flat = counterfactual_flat * 100

    if obs > factual_flat.max():
        obs = obs * 0.67

    out = plot_FUN(factual_flat, counterfactual_flat, obs, plot_name = plot_name, *args, **kw)    
    if add_legend:
        plt.legend()

    return out

def plot_attribution_scatter(regions, figname, *args, **kw):
    metrics = ["mean", 'pc-95.0']
    fig, axes = plt.subplots(len(regions), 2, figsize=(12, len(regions)*3.5))
    out = []
    for i, metric in enumerate(metrics):
        outi = []
        for j, region in enumerate(regions):
            ax = axes[j, i]
            print(region)
            outi.append(plot_for_region(region, metric, plot_fact_vs_ratio, ax = ax, 
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
    
    dir1 = "outputs/outputs_scratch/ConFLAME_nrt-attribution//"
    dir2 = "-2425/time_series/_18-frac_points_0.5/"



    regions = ["Amazon", "Congo", "Pantanal"]
    #retgions = {key: regions_info[key] for key in region_names if key in regions_info}
    obs_dir = 'data/data/driving_data2425//'
    obs_file = 'burnt_area_data.csv'
    
    outs_era5 = plot_attribution_scatter(regions, "attribution_scatter_era5_2425",
                             dir1 = dir1, dir2 = dir2,
                             obs_dir = obs_dir, obs_file = obs_file) 
    
    dir1 = "outputs/outputs/ConFLAME_"
    dir2 = "-2425-attempt12/time_series/_15-frac_points_0.5/"

    outs_isimip = plot_attribution_scatter(regions, 
                             "attribution_scatter_isimip_2425",
                             dir1 = dir1, dir2 = dir2,
                             obs_dir = obs_dir, obs_file = obs_file, all_mod_years = True) 

    outs_human = plot_attribution_scatter(regions, 
                             "attribution_scatter_isimip_human_2425",
                             dir1 = dir1, dir2 = dir2,
                             obs_dir = obs_dir, obs_file = obs_file, 
                             factual_name = "counterfactual", 
                             counterfactual_name = "early_industrial",
                             all_mod_years = True) 

    outs_combined = []
    for ii in range(len(outs_era5)):
        outi = []
        for jj in range(len(outs_era5[ii])):
            outi.append(np.random.choice(outs_era5[ii][jj], 1000) + \
                np.random.choice(outs_isimip[ii][jj], 1000)/2.0)
        outs_combined.append(outi)#set_trace()

    
    f = open('temp/store.pckl', 'wb')
    pickle.dump([outs_era5, outs_isimip, outs_combined, outs_human], f)
    f.close()
    
    f = open('temp/store.pckl', 'rb')
    outs_era5, outs_isimip, outs_combined, outs_human = pickle.load(f)
    f.close()

    # Example labels for the sources
    sources = ['Climate (HadGEM+ERA5)', 'Climate (ISIMIP3a)', 'Climate (Combined)', 'Human']

    # Group data
    all_sources = [outs_era5, outs_isimip, outs_combined, outs_human]

    # Flatten into long-form dataframe for seaborn
    records = []
    for region_idx, region in enumerate(regions):
        for source_idx, source_name in enumerate(sources):
            for kind_idx, kind in enumerate(["Mean", "Extreme"]):  # 0 = mean, 1 = extreme
                samples = all_sources[source_idx][kind_idx][region_idx]
                for val in samples:
                    records.append({
                        'Region': region,
                        'Source': source_name,
                        'Impact Type': kind,
                        'Relative Change (%)': val
                    })
    
    df = pd.DataFrame.from_records(records)
    
    # Set up the plot
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top: Mean
    sns.violinplot(
        data=df[df["Impact Type"] == "Mean"],
        x="Region", y="Relative Change (%)", hue="Source",
        split=False, inner="quartile", palette="Set2", ax=axes[0]
    )
    axes[0].set_title("Mean Burnt Area Impact")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].legend(loc="lower left")
    axes[0].axhline(0, color='k', linestyle='--')
    # Bottom: Extreme
    sns.violinplot(
        data=df[df["Impact Type"] == "Extreme"],
        x="Region", y="Relative Change (%)", hue="Source",
        split=False, inner="quartile", palette="Set2", ax=axes[1]
    )
    axes[1].set_title("Extreme Burnt Area Impact")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].legend(loc="lower left")
    axes[0].axhline(0, color='k', linestyle='--')

    # Tidy up
    plt.tight_layout()
    plt.show()
    set_trace()

import pathlib
import pandas as pd
import numpy as np
from pdb import set_trace
from datetime import datetime
# Define the directory path

import matplotlib.pyplot as plt
import sys
sys.path.append('SoW_info/')
from state_of_wildfires_colours  import SoW_cmap

def obtain_mnths(dat, target_mnths, target_years):
    try:
        times = dat['time']
    except:
        times = dat.index
    try:
        mnths = [int(tm.split('-')[1]) for tm in times]
    except:
        set_trace()
    years = [int(tm.split('-')[0]) for tm in times]

    which_mnths = [mnth in target_mnths and yr in target_years\
                   for mnth, yr in zip(mnths, years)]
    return which_mnths
    
def get_mod_dat_for_years_mnths(path, target_mnths = range(0, 12),
                                yrs_range = [2010, 2100]):
    dat = pd.read_csv(path).T.iloc[1:]
    dat0 = dat.copy()
    syrs = range(max([int(dat.index[ 0].split('-')[0])  , yrs_range[0]]),
                 min([int(dat.index[-1].split('-')[0])+1, yrs_range[1]]))
    
    def get_mod_dat_for_year(syr):
        which_mnths = obtain_mnths(dat, target_mnths, range(syr, syr + 1))
        ydat = dat.loc[which_mnths]
        BA = dat.loc[which_mnths].mean()
        return(BA)

    
    dat = np.array([get_mod_dat_for_year(syr) for syr in syrs])
    try:
        out = pd.DataFrame(data=dat.T, columns=syrs)
    except:
        set_trace()
    return out

def open_BA_projections(models, periods, mod_path,
                        sample_type = "mean", run = "Evaluate.csv", 
                        file_path = "/members/absolute/"):

    def BA_for_period_mod(period, model):
        path = mod_path + '/' + period + '/' +  model + '/' +  sample_type + '/' +  \
            file_path + '/' +  run
        
        return get_mod_dat_for_years_mnths(path, target_mnths)

    def BA_for_model(model):
        out = [BA_for_period_mod(period, model) for period in periods]
        return [pd.concat([out[0], ot], axis=1) for ot in out[1:]]
    
    return [BA_for_model(model) for model in models]

def plot_projections(ax, sample_type = "mean", run = "Evaluate.csv", obs = None, 
                     cmap = 'gradient_hues', *args, **kw):
    print(run)
    BAs = open_BA_projections(sample_type = sample_type, run = run, *args, **kw) 
    
    if obs.shape == ():
        mask = np.array(BAs) > (obs/ 100.0)
        occurance = np.mean(mask, axis = 2)
    else:
        occurance = np.where(obs, np.array(BAs), np.nan)
        occurance = 100*np.nanmean(occurance, axis = 2)
        mask = obs
    *other_dims, last_dim = occurance.shape
    block_size = 10
    num_blocks = last_dim // block_size
    decade_occurance = occurance.reshape(*other_dims, num_blocks, block_size).mean(axis=-1)  
    #set_trace()
    a, b, c = decade_occurance.shape
    
    # 1. Define spacing
    bar_width = 0.2
    group_gap = 0.5  # Gap between different 'c' groups
    x_base = np.arange(c) * (b * bar_width + group_gap) 

    #fig, ax = plt.subplots(figsize=(10, 6))
    decade_occurance0 = decade_occurance[:, :, 0].copy()
    cols = SoW_cmap[cmap](np.linspace(0, 1, b+1))
    ax.yaxis.grid(True)
    for i in range(c):
        if obs.shape == ():
            decade_occurance[:, :, i] /= decade_occurance0
        else: 
            decade_occurance[:, :, i] -= decade_occurance0
            
        for j in range(b):
            # Calculate x-position for this specific bar
            x_pos = x_base[i] + (j * bar_width)
            
            # Extract data for this bar: shape (a,)
            bar_values = decade_occurance[:, j, i]
            
            # Draw the main bar spanning the range of 'a'
            #set_trace()
            if i == 0:
                col = cols[0]
            else:
                col = cols[j+1]

            current_label = periods[j + 1] if i == 1 else None

            ax.bar(x_pos, bar_values.max() - bar_values.min(), 
                   bottom=bar_values.min(), width=bar_width, 
                   color=col, alpha = 0.6, edgecolor='black', label=current_label)
            
            # Draw each point in 'a' as a thin horizontal line
            ax.hlines(y=bar_values, xmin=x_pos - bar_width/2, color=col,
                      xmax=x_pos + bar_width/2, linewidth=1)
     
    ax.set_xticks(x_base + (b-1)*bar_width/2)
    ax.set_xticklabels([f'{yr}s' \
                       for yr in range(yrs_range[0], yrs_range[1], 10)])
    ax.tick_params(axis='x', rotation=45)
    
    return mask


def plot_future_projections(obs_path, mod_path, Region_name, periods, models, 
                            sample_types, sample_names, runs, plot_names, cmaps, 
                            target_mnths, target_years, yrs_range):
    obs = pd.read_csv(obs_path)
    which_mnths = obtain_mnths(obs, target_mnths, target_years)
    obs0 = np.mean(obs['mean_burnt_area'][which_mnths])
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 7), sharex=True, layout='constrained')
    axes = axes.T.flatten()
    axi = 0
    for sample_type, sample_name in zip(sample_types, sample_names):
        obs = obs0.copy()
        for run, title, cmap in zip(runs, plot_names, cmaps):
            ax = axes[axi]
            obs = plot_projections(ax, sample_type, run, obs, cmap, 
                                   models, periods, mod_path)
            if run == runs[0]:
                ax.set_title(sample_name)
            if sample_type == sample_types[0]:
                ax.set_ylabel(title)
                ax.legend()
            axi += 1

    fig.suptitle(Region_name, fontsize=16)
    fig.savefig("figs/futures" + Region_name + ".png", dpi = 300)

if __name__=="__main__":
    
    obs_path = "data/data/driving_data_base/Amazon/burnt_area_data.csv"
    mod_path = "outputs/outputs_scratch/Base-ISIMIP_large-4/ConFLAME_Amazon-2425/time_series/_15-frac_points_0.5/"
    Region_name = "Amazonia"
    periods = ["historical", "ssp126", "ssp370", "ssp585"]
    models  = ["GFDL-ESM4-", "IPSL-CM6A-LR-", "MPI-ESM1-2-HR-", "UKESM1-0-LL-"]
    sample_types = ["mean", "pc-90.0", "pc-95.0"]
    sample_names = ["All region", "Top 10% Burned Area", "Top 5% Burned Area"]
    runs = ["Evaluate.csv", "standard-Fuel.csv", 
           "standard-Moisture.csv"]
    plot_names = ["Burned Area\n(Times more likely)", "Fuel control\nstrength change (%)", "Dryness controll\nstrength change (%)"]
    cmaps = ['gradient_hues', "gradient_teal", "gradient_red"]
    target_mnths = range(5, 7)
    target_years = [2024]
    yrs_range = [2010, 2100]
    
plot_future_projections(obs_path, mod_path, Region_name, periods, models, 
                        sample_types, sample_names, runs, plot_names, cmaps, 
                        target_mnths, target_years, yrs_range)

obs_path = "data/data/driving_data_base/Pantanal/burnt_area_data.csv"
mod_path = "outputs/outputs_scratch/Base-ISIMIP_large-4/ConFLAME_Pantanal-2425/time_series/_15-frac_points_0.5/"
Region_name = "Pantanal"

plot_future_projections(obs_path, mod_path, Region_name, periods, models, 
                        sample_types, sample_names, runs, plot_names, cmaps, 
                        target_mnths, target_years, yrs_range)   

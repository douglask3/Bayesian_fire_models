import pathlib
import pandas as pd
import numpy as np
from pdb import set_trace
from datetime import datetime
# Define the directory path


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
                                syrs = range(2010, 2099)):
    dat = pd.read_csv(path).T.iloc[1:]
    def get_mod_dat_for_year(syr):
        which_mnths = obtain_mnths(dat, target_mnths, range(syr, syr + 1))
        ydat = dat.loc[which_mnths]
        BA = dat.loc[which_mnths].mean()
        return(BA)

    dat = [get_mod_dat_for_year(syr) for syr in syrs]
    np.array(dat)

if __name__=="__main__":
    hist_path = "outputs/outputs_scratch/Base-ISIMIP_large-4/ConFLAME_Amazon-2425/time_series/_15-frac_points_0.5/historical/GFDL-ESM4-/mean/members/absolute/Evaluate.csv"

    obs_path = "data/data/driving_data_base/Amazon/burnt_area_data.csv"

    target_mnths = range(5, 7)
    target_years = [2024]

    obs = pd.read_csv(obs_path)
    which_mnths = obtain_mnths(obs, target_mnths, target_years)
    obs = np.mean(obs['mean_burnt_area'][which_mnths])

    mod = get_mod_dat_for_years_mnths(hist_path, target_mnths)
    set_trace()
    
    
    
    

import pandas as pd
import sys
sys.path.append('libs/')
from climtatology_difference import *
from pdb import set_trace

def for_region(dir1, dir2, region, controls, control_name):
    mod_dir = dir1 + region + dir2
    def open_control(control):
        mod_file = mod_dir + "/absolute/" + control + ".csv"
        df = pd.read_csv(mod_file)
        df.set_index(df.columns[0], inplace=True)
        return df
    
    control_df = open_control(controls[0])
    for cntr in controls[1:]:
        control_df *= open_control(cntr)
    
    control_df = control_df**(1/len(controls))
    control_df.to_csv(mod_dir + "/absolute/" + control_name + ".csv")
    
    climatology, anomaly, ratio = climtatology_difference(control_df.values[:,1:])

    control_df.iloc[:,1:] = anomaly
    control_df.to_csv(mod_dir + "/anomaly/" + control_name + ".csv")
    
    control_df.iloc[:,1:] = ratio
    control_df.to_csv(mod_dir + "/ratio/" + control_name + ".csv")    
    
dir1  = "outputs/outputs_scratch/ConFLAME_nrt-drivers6/" 
dir2s = ["-2425/time_series/_21-frac_points_0.5/baseline-/mean/members/",
        "-2425/time_series/_21-frac_points_0.5/baseline-/pc-95.0/members/"]

controls = [['potential_climateology-Fuel', 'potential_climateology-Moisture'], 
            ['potential_climateology-Weather', 'potential_climateology-Wind'],
            ['potential_climateology-Suppression', 'potential_climateology-Ignition']]
control_names = ['potential_climateology-Fuel-Moisture', 'potential_climateology-Weather-Wind', 
                 'potential_climateology-Suppression-Ignition']

controls = [['standard-Fuel', 'standard-Moisture'], 
            ['standard-Weather', 'standard-Wind'],
            ['standard-Suppression', 'standard-Ignition']]
control_names = ['standard-Fuel-Moisture', 'standard-Weather-Wind', 
                 'standard-Suppression-Ignition']
regions = ['Amazon', 'Pantanal', 'LA',  'Congo']

for region in regions:
    for dir2 in dir2s:
        for cnrts, control_name in zip(controls, control_names):
            for_region(dir1, dir2, region, cnrts, control_name)


import sys
sys.path.append('.')
sys.path.append('plotting/')
sys.path.append('src/')

from BA_and_controls_ts import *



if __name__=="__main__":
    dir1 = "outputs/outputs_scratch/ConFLAME_nrt-drivers-base-4/" 
    dir2 = ["/time_series/_21-frac_points_0.05/baseline-/mean/members/",
            "/time_series/_21-frac_points_0.05/baseline-/pc-95.0/members/"]
    obs_dir = "data/data/driving_data_base/"

    regions = ['Pantanal']
    regions = ['Amazon']
    region_sizes = [1063.64005]
    region_sizes = [3592.46844]
    region_names = ['Pantanal']
    region_names = ['Amazonia']
    
    run_regions_controls(regions, region_sizes, controls_potential, contol_names, dir1, dir2, obs_dir, 'ratio')
    run_regions_controls(regions, region_sizes, controls_potential, contol_names, dir1, dir2, obs_dir, 'anomaly')
    run_regions_controls(regions, region_sizes, controls_potential, contol_names, dir1, dir2, obs_dir, 'absolute')

    region_dates = [[pd.Timestamp(datetime(2022, 12, 15)), pd.Timestamp(datetime(2025, 3, 15))],
               [pd.Timestamp(datetime(2022, 12, 15)), pd.Timestamp(datetime(2025, 3, 15))],
               [pd.Timestamp(datetime(2022, 12, 15)), pd.Timestamp(datetime(2025, 3, 15))],
               [pd.Timestamp(datetime(2022, 12, 15)), pd.Timestamp(datetime(2025, 3, 15))]]





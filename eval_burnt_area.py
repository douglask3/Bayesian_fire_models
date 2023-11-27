from eval_trends import *

if __name__=="__main__":  
    dir_isimip3a = "/scratch/hadea/isimip3a/u-cc669_isimip3a_fire/"
    dir_isimip2b = "/hpc/data/d05/cburton/jules_output/u-cf137/"
    filenames_model = [dir_isimip3a + "20CRv3-ERA5_obsclim/jules-inferno-vn6p3_20crv3-era5_obsclim_histsoc_default_burntarea-total_global_monthly_1901_2021.nc",
                       dir_isimip3a + "20CRv3_obsclim/jules-inferno-vn6p3_20crv3_obsclim_histsoc_default_burntarea-total_global_monthly_1901_2015.nc",
                       dir_isimip3a + "20CRv3-W5E5_obsclim/jules-inferno-vn6p3_20crv3-w5e5_obsclim_histsoc_default_burntarea-total_global_monthly_1901_2019.nc",
                       dir_isimip3a + "GSWP3-W5E5_obsclim/jules-inferno-vn6p3_gswp3-w5e5_obsclim_histsoc_default_burntarea-total_global_monthly_1901_2019.nc",
                       dir_isimip2b + "GFDL-ESM2M/*.ilamb.*.nc",
                       dir_isimip2b + "HADGEM2-ES/*.ilamb.*.nc",
                       dir_isimip2b + "IPSL-CM5A-LR/*.ilamb.*.nc",
                       dir_isimip2b + "MIROC5/*.ilamb.*.nc"]
                       
    names_model = ["isimip3a-20CRv3-ERA5_obsclim",
                   "isimip3a-20CRv3_obsclim",
                   "isimip3a-20CRv3-W5E5_obsclim",
                   'isimip3a-era5-obsclim',
                   "isimip2b-GFDL-ESM2M",
                   "isimip2b-HADGEM2-ES",
                   "isimip2b-IPSL-CM5A-LR",
                   "isimip2b-MIROC5"]
    
    variable_model = 'burnt_area_gb'

    dir_observation = "/data/dynamic/dkelley/fireMIPbenchmarking/data/benchmarkData/"
    filenames_observation = ["ISIMIP3a_obs/GFED4.1s_Burned_Fraction.nc", \
                             "ISIMIP3a_obs/FireCCI5.1_Burned_Fraction.nc", \
                             "ISIMIP3a_obs/GFED500m_Burned_Percentage.nc"]
    filenames_observation = [dir_observation + file for file in filenames_observation]
    
    observations_names = ['GFED4.1s', 'FireCCI5.1', 'GFED500m']

    year_range = [1996, 2020]
    n_itertations = 1000
    tracesID = 'burnt_area_trace'
    mod_scale = 1.0/100.0
    obs_scale = [1.0, 1.0, 1.0/100.0]

    units = '1'
    output_dir = 'outputs/burnt_area_new'

    region_type = 'gfed'

    run_all_eval_for_models(filenames_model, names_model, variable_model,
                            filenames_observation, observations_names,
                            year_range, 
                            n_itertations, tracesID, mod_scale,  obs_scale, units,
                            output_dir, region_type = region_type)

from eval_trends import *

if __name__=="__main__":    
    filenames_model = ["/hpc/data/d05/cburton/jules_output/u-cf137/GFDL-ESM2M/*.ilamb.*.nc",
                      "/hpc/data/d05/cburton/jules_output/u-cf137/HADGEM2-ES/*.ilamb.*.nc",
                      "/hpc/data/d05/cburton/jules_output/u-cf137/IPSL-CM5A-LR/*.ilamb.*.nc",
                      "/hpc/data/d05/cburton/jules_output/u-cf137/MIROC5/*.ilamb.*.nc", 
                      "/scratch/hadea/isimip3a/u-cc669_isimip3a_fire/20CRv3-ERA5_obsclim/jules-vn6p3_20crv3-era5_obsclim_histsoc_default_burntarea-total_global_monthly_1901_2021.nc"]

    names_model = ["isimip2b-GFDL-ESM2M",
                   "isimip2b-HADGEM2-ES",
                   "isimip2b-IPSL-CM5A-LR",
                   "isimip2b-MIROC5",
                   'isimip3a-era5-obsclim']
    
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
    output_file = 'outputs/trend_burnt_area_metric_results'
    output_maps = 'outputs/burnt_area/'

    region_type = 'gfed'

    run_all_eval_for_models(filenames_model, names_model, variable_model,
                            filenames_observation, observations_names,
                            year_range, 
                            n_itertations, tracesID, mod_scale,  obs_scale, units,
                            output_file, output_maps, region_type = region_type)

from eval_trends import *

if __name__=="__main__":    
    filenames_model = ["/hpc/data/d05/cburton/jules_output/u-cf137/GFDL-ESM2M/*.ilamb.*.nc",
                      "/hpc/data/d05/cburton/jules_output/u-cf137/HADGEM2-ES/*.ilamb.*.nc",
                      "/hpc/data/d05/cburton/jules_output/u-cf137/IPSL-CM5A-LR/*.ilamb.*.nc",
                      "/hpc/data/d05/cburton/jules_output/u-cf137/MIROC5/*.ilamb.*.nc"]

    names_model = ["isimip2b-GFDL-ESM2M",
                   "isimip2b-HADGEM2-ES",
                   "isimip2b-IPSL-CM5A-LR",
                   "isimip2b-MIROC5"]
    
    variable_model = ['npp_n_gb', '-resp_s_to_atmos_gb', '-WP_fast_out', '-WP_med_out', '-WP_slow_out', '-harvest_gb', '-veg_c_fire_emission_gb', '-burnt_carbon_rpm', '-burnt_carbon_dpm']
    
    dir_observation = "/scratch/cburton/scratch/ISIMIP_PAPER/BenchmarkNBP/"
    filenames_observation = ["carbonTracker_360x180/CT2019_nbp.nc", \
                             "cams_360x180/cams73_latest_co2_flux_surface_mm_360x180_taxis.nc", \
                             "CarboScope_360x180/s99oc_v2020_co2fluxLand_monthly_360x180_taxis.nc"]
    filenames_observation = [dir_observation + file for file in filenames_observation]
    
    observations_names = ['CT2019', 'cams73', '"s99oc_v2020']


    year_range = [2001, 2017]
    n_itertations = 1000
    tracesID = 'nbp_test-with_fire'
    mod_scale = 1/12
    obs_scale = [86400*12/1000*360*(-1)/12, 1000/30/1000*360*(-1)/12, 360*-1/1000/12]

    units = '1'
    output_file = 'outputs/nbp_test-with_fire'
    output_maps = 'outputs/NBP/'

    region_type = 'gfed'

    run_all_eval_for_models(filenames_model, names_model, variable_model,
                            filenames_observation, observations_names,
                            year_range, 
                            n_itertations, tracesID, mod_scale,  obs_scale, units,
                            output_file, output_maps, region_type = region_type)

    
    jules_out_dir = "/hpc/data/d01/hadcam/jules_output/ALL_u-bk886_isimip_0p5deg_origsoil_dailytrif/"
    filenames_model = [jules_out_dir + "/GFDL-ESM2M/*.ilamb.*.nc",
                       jules_out_dir + "/HADGEM2-ES/*.ilamb.*.nc",
                       jules_out_dir + "/IPSL-CM5A-LR/*.ilamb.*.nc",
                       jules_out_dir + "/MIROC5/*.ilamb.*.nc"]

    variable_model = ['npp_n_gb', '-resp_s_to_atmos_gb', '-WP_fast_out', '-WP_med_out', '-WP_slow_out', '-harvest_gb']

    tracesID = 'nbp_test-without_fire'
    output_file = 'outputs/nbp_test-withouts_fire'


    run_all_eval_for_models(filenames_model, names_model, variable_model,
                            filenames_observation, observations_names,
                            year_range, 
                            n_itertations, tracesID, mod_scale,  obs_scale, units,
                            output_file, output_maps, region_type = region_type)
    

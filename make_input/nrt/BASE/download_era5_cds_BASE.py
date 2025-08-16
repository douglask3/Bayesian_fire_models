import sys
sys.path.append('.')
sys.path.append('make_input/nrt/')
from download_era5_cds import *
from pdb import set_trace 

if __name__=="__main__":  
    temp_dir = "/data/users/douglas.kelley/Bayesian_fire_models/data-cds/era5_nrt/"
    out_dir = "data/data/driving_data2425/nrt_attribution//"

    shapefile_path = "data/BASE_shapes/"
    region_names = ["Amazon", "Pantanal"]
    
    for region in region_names:
        shapefile = shapefile_path + '/' + region + '/' + region + '.shp'
        for years in yearss:
            download_era5(variables, years, months = range(12), 
                          yr_now = yr_now, mnth_now = mnth_now,
                          area = area, region_name = region,
                          dataset = dataset, 
                          out_dir = out_dir, 
                          temp_dir = temp_dir,
                          shapefile_path = shapefile,
                          region_in_shapefile = False)

import os.path
import sys
sys.path.append('.')
sys.path.append('libs/make_input/nrt/')
from HadGEM_extract import *


if __name__=="__main__":
    temp_dir = "/data/scratch/douglas.kelley/Bayesian_fire_models/temp/hadgem_nrt2/BASE/"
    out_dir = "data/data/driving_data_base/nrt_attribution/"
    
    shapefile_path = "data/BASE_shapes/"
    region_names = ["Amazon", "Pantanal"]

    for start_year in start_years:
        for rname in region_names:
            shapefile = shapefile_path + '/' + rname + '/' + rname + '.shp'
            process_variables(experiments, variables, start_year, dir, sub_dir,
                              out_dir, 
                              temp_dir,
                              region_names = [rname],
                              shapefile_path = shapefile,
                              region_in_shapefile = False)

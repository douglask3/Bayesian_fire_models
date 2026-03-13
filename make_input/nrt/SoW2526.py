import sys
sys.path.append('../../libs/')
sys.path.append('libs/')
sys.path.append('make_inputs/nrt/')

from download_era5_cds import *
from HadGEM_extract import *


if __name__=="__main__":  
    data_dir = "data/data/driving_data2526/"
    shapefile_path = data_dir + "/Focal_regions/SoW2526_Focal_MASTER_20260218.shp"
    out_dir = data_dir + "/nrt_raw/"
    region_names = ["Northwest Iberia",
                    "Midwestern Canadian Shield forests",
                    "Chilean Temperate Forests and Matorral",
                    "Scottish Highlands",
                    "Southeast South Korea"]
    run_for_report(shapefile_path, region_names,  out_dir)
    set_trace()
    temp_dir = data_dir + "/hadgem_nrt2/"

    for start_year in start_years:
        process_variables(experiments, variables, start_year, dir, sub_dir,
                          out_dir, 
                          temp_dir,
                          region_names = region_names,
                          shapefile_path = shapefile_path)
    

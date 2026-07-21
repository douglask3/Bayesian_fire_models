import sys
sys.path.append('../../libs/')
sys.path.append('libs/')
sys.path.append('make_inputs/nrt/')

from download_era5_cds import *
from HadGEM_extract import *
from make_er5_extra_vars import *
from regrid_hyde import *

if __name__=="__main__":  
    data_dir = "data/data/driving_data2526/"
    shapefile_path = data_dir + "/Focal_regions/SoW2526_Focal_MASTER_20260218.shp"
    out_dir = data_dir + "/nrt_raw/"
    region_names = [
                    #"Midwestern Canadian Shield forests",
                    #"Chilean Temperate Forests and Matorral",
                    "Northwest Iberia",
                    #"Scottish Highlands",
                    #"Southeast South Korea"
                    ]

           
    #set_trace()
    #run_era5_download_for_report(shapefile_path, region_names,  out_dir)
    
    #make_era5_extra_vars(data_dir + '/nrt_raw/', region_names)
    
    temp_dir = data_dir + "/hadgem_nrt2/"
    
    for start_year in start_years:
        process_variables(experiments, variables, start_year, dir, sub_dir,
                          out_dir, 
                          temp_dir,
                          region_names = region_names,
                          shapefile_path = shapefile_path)
    
    #regrid_hyde_for_regions(region_names, data_dir) 
    

import sys
sys.path.append('../../libs/')
sys.path.append('libs/')
sys.path.append('make_inputs/nrt/')

from download_era5_cds import *


if __name__=="__main__":  
    data_dir = "data/data/driving_data2425"
    shapefile_path = data_dir + "/SoW2425_shapes/SoW2526_Focal_MASTER_20260218.shp"
    out_dir = data_dir + "nrt_raw/"
    region_names = ["Northwest Iberia",
                    "Midwestern Canadian Shield forests",
                    "Chilean Temperate Forests and Matorral",
                    "Scottish Highlands",
                    "Southeast South Korea"]
    run_for_report(shapefile_path, region_names,  out_dir)

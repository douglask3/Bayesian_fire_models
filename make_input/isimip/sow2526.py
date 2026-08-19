import sys
sys.path.append('.')
sys.path.append('make_input/isimp/')
from regrid_all import *
from landuse_change import *
from pdb import set_trace
 
def run_for_report(region_names, output_dir, shp_filename):
    subset_functions_main = [contrain_to_sow_shapefile]
    vcf_dir = "same"
    
    for region_name in region_names:
        subset_function_argss_main = [{'shp_filename': shp_filename, 
                                       'name': region_name}]
        for_region(subset_functions_main, subset_function_argss_main, 
                   vcf_dir, region_name = region_name.replace(' ', '_'), 
                   output_dir = output_dir)
    run_LULCC_for_all_regions(region_names, output_dir)

if __name__=="__main__":

    output_dir = "data/data/driving_data2526/"
    region_names = ["Northwest Iberia",
                    "Midwestern Canadian Shield forests",
                    "Chilean Temperate Forests and Matorral",
                    "Scottish Highlands",
                    "Southeast South Korea"
                    ]
    shapefile_path = output_dir + "/Focal_regions/SoW2526_Focal_MASTER_20260218.shp"
    
    run_for_report(region_names, output_dir, shapefile_path)


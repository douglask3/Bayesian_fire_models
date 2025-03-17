import sys

sys.path.append('.')
sys.path.append('make_input/isimp/')
from regrid_all import *


output_dir = "data/data/driving_data2425/"
if __name__=="__main__":

    shape_names = ['northeast India', "Alberta",
                   "Los Angeles",
                   "Congo basin",
                   "Amazon and Rio Negro rivers",
                   "Pantanal"]
    region_names = [ 'NWIndia', 'Alberta', 'LA', 'Congo', 'Amazon', 'Pantanal']
    shp_filename = "data/data/SoW2425_shapes/SoW2425_Focal_MASTER_20250221.shp"
    subset_functions_main = [contrain_to_sow_shapefile]
    vcf_dir = "same"
    
    for shape_name, region_name in zip(shape_names, region_names):
        subset_function_argss_main = [{'shp_filename': shp_filename, 'name': shape_name}]
        for_region(subset_functions_main, subset_function_argss_main, 
                   vcf_dir, region_name = region_name, output_dir = output_dir)


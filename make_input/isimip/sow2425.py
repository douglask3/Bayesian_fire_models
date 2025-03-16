import sys

sys.path.append('.')
sys.path.append('make_input/isimp/')
from regrid_all import *


output_dir = "data/data/driving_data2425/"
if __name__=="__main__":
    shp_filename = "data/data/SoW2425_shapes/SoW2425_Focal_MASTER_20250221.shp"
    subset_functions_main = [contrain_to_sow_shapefile]
    subset_function_argss_main = [{'shp_filename': shp_filename, 'name': 'Pantanal'}]
    vcf_dir = "same"
    for_region(subset_functions_main, subset_function_argss_main, 
               vcf_dir, region_name = 'Pantanal', output_dir = output_dir)


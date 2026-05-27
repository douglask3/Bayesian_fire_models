import sys

sys.path.append('.')
sys.path.append('make_input/isimp/')
from regrid_all import *


output_dir = "data/data/driving_data_base/"
if __name__=="__main__":

    shape_names = ["Amazon", "Pantanal"]
    shp_dir = "data/BASE_shapes/"
    subset_functions_main = [contrain_to_shapefile]
    vcf_dir = "same"
    
    for shape_name in shape_names:
        shp_filename = shp_dir + shape_name + '/' + shape_name + '.shp' 
        subset_function_argss_main = [{'shp_filename': shp_filename, 'name': shape_name}]
        for_region(subset_functions_main, subset_function_argss_main, 
                   vcf_dir, region_name = shape_name, output_dir = output_dir)


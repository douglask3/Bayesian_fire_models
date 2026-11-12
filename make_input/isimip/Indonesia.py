import sys
sys.path.append('.')
sys.path.append('make_input/isimp/')
from regrid_all import *
from landuse_change import *
from pdb import set_trace
#32226346 

if __name__=="__main__":

    output_dir = "data/data/driving_data2526/"
    region_name = "Indonesia"
    subset_functions_main = [constrain_natural_earth]
    subset_function_argss_main = [{'Country': 'Indonesia'}]
    years = [[1900 + i, 1909 + i] for i in range(0, 120, 10)]
    hist_years = [[1994, 2014]]
    futr_years = [[2015, 2019]] + [[2020 + i, 2029 + i] for i in range(0, 80, 10)]
    
    shapefile_path = None
    vcf_dir = "same"
    
    for_region(subset_functions_main, subset_function_argss_main, 
                   vcf_dir, region_name = region_name.replace(' ', '_'), 
                   output_dir = output_dir,
                   years = years, hist_years = hist_years, futr_years = futr_years)

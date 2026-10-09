import sys
sys.path.append('.')
sys.path.append('make_input/isimp/')
from regrid_all import *
from landuse_change import *
from pdb import set_trace
 

if __name__=="__main__":

    output_dir = "data/data/driving_data2526/"
    region_names = ["Global"]
    years = [[1900 + i, 1909 + i] for i in range(0, 120, 10)]
    years = years[-2:0]
    hist_years = [[1994, 2014]]
    futr_years = [[2015, 2019]] + [[2020 + i, 2029 + i] for i in range(0, 80, 10)]
    futr_years = futr_years[0:2]
    shapefile_path = None
     
    run_for_report(region_names, output_dir, shapefile_path, 
                   years = years, hist_years = hist_years, futr_years = futr_years)


